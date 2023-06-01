# Goal Conditioned Behavior Cloning (GCBC) Agent

import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from fluidlab.models.dataloader import NumPyTrajectoryDataset
import itertools
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


class ConvModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, pad):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel,
            padding=pad,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MLPModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.l1(x)))


class GCBCAgent(nn.Module):
    def build_conv_layers_(self) -> nn.Sequential:
        # We will take in a concatenation of the current scene obs and a goal configuration
        conv_module_one = ConvModule(6, 32, 5, 1, 1)
        conv_module_two = ConvModule(32, 64, 3, 1, 0)
        conv_module_three = ConvModule(64, 64, 3, 3, 0)
        return nn.Sequential(conv_module_one, conv_module_two, conv_module_three)

    def build_mlp_layers_(self) -> nn.Sequential:
        flatten = nn.Flatten()
        l1 = MLPModule(64 * 84 * 84, 128)
        l2 = MLPModule(128, 128)
        l3 = MLPModule(
            128, self.ac_dim
        )  # output a mean vector corresponding to the number of action components
        return nn.Sequential(flatten, l1, l2, l3)

    def __init__(self, ac_dim):
        super().__init__()
        self.ac_dim = ac_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logstd = nn.Parameter(torch.zeros((ac_dim), device=self.device))
        self.conv_layers = self.build_conv_layers_()
        self.mlp_layers = self.build_mlp_layers_()

    def forward(self, x) -> distributions.Distribution:
        means = self.mlp_layers(self.conv_layers(x)).to(self.device)
        return distributions.Normal(means, torch.exp(self.logstd))

    def test_forward(self):
        # randomly generated input
        x = torch.rand(2, 6, 256, 256)
        dist = self(x)
        return dist.sample()

    def update(self, obs_no, ac_na):
        dist = self.forward(obs_no)
        pred_ac_na = dist.sample()
        return pred_ac_na.to(self.device)


def train(out_weights_file):
    train = NumPyTrajectoryDataset("npy_trajs", train=True)
    
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=64, shuffle=True, num_workers=2
    )
    agent = GCBCAgent(3)
    agent.load_state_dict(torch.load("fluidlab/models/weights/gcbc_weights_no_padded_acs.pt"))
    gcbc_optim = optim.Adam(
        itertools.chain(
            agent.conv_layers.parameters(),
            agent.mlp_layers.parameters(),
            [agent.logstd],
        )
    )
    for batch_idx, (img_obs, img_obs_next, action, _) in enumerate(train_loader):
        inpt = torch.cat((img_obs, img_obs_next), 1)
        inpt = inpt.type('torch.FloatTensor')
        dist = agent.forward(inpt)
        action = action.detach().to(device)
        log_probs = dist.log_prob(action)
        loss = -torch.sum(log_probs, dim=1).mean()
        wandb.log({"train_loss": loss})
        wandb.log({'log_prob': -loss})
        if loss < 0.3:
            torch.save(agent.state_dict(), out_weights_file)
            return
        loss.backward()
        if batch_idx % 100 == 0:
            torch.save(agent.state_dict(), out_weights_file)
        gcbc_optim.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_weights_file", type=str, default="gcbc_weights.pt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    run = wandb.init(
        project="gcbc-training-fluid-manip"
    )
    train(out_weights_file=args.out_weights_file)
