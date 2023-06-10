import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fluidlab.models.vae import VAE
from fluidlab.models.dataloader import NumPyTrajectoryDataset
import argparse


class FCModule(nn.Module):
    def __init__(self, in_dim, out_dim, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        self.l1 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return x


class VAEDynamicsModel(nn.Module):
    def __init__(self, n_latent_features, action_dim, device):
        super().__init__()
        self.input_latent_features = n_latent_features
        self.device = device
        self.action_dim = action_dim
        self.fc1 = FCModule(self.input_latent_features + self.action_dim, 512)
        self.fc2 = FCModule(512, 256)
        self.fc3 = FCModule(256, 128)
        self.fc4 = FCModule(128, 32, use_activation=False)

    def forward(self, z_obs, action):
        z_obs = z_obs.to(self.device).type("torch.cuda.FloatTensor")
        action = action.to(self.device).type("torch.cuda.FloatTensor")
        # concat latent vector and action
        if len(z_obs.size()) != 2:
            z_obs = z_obs.unsqueeze(0)
        if len(action.size()) != 2:
            action = action.unsqueeze(0)
        x = torch.cat((z_obs, action), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_vae_weights", type=str)
    parser.add_argument("--in_trajs_file", type=str, default="low_var_trajs")
    parser.add_argument("--out_weights", type=str)
    parser.add_argument("--ac_dim", type=int, default=3)
    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, in_vae_weights, in_trajs_file, ac_dim) -> None:
        self.in_vae_weights = in_vae_weights
        self.in_trajs_file = in_trajs_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ac_dim = ac_dim
        self.vae = VAE("a", "a", standalone=False).to(self.device)
        self.vae.load_state_dict(torch.load(self.in_vae_weights))
        self.train_data = NumPyTrajectoryDataset(trajs_dir=in_trajs_file, train=True) 
        self.val_data = NumPyTrajectoryDataset(trajs_dir=in_trajs_file, train=False)
        self.train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_data, batch_size=64, shuffle=True, num_workers=2)
        self.vae_dyn = VAEDynamicsModel(self.vae.n_latent_features, self.ac_dim, self.device).to(self.device)
        self.vae_dyn_optim = optim.Adam(self.vae_dyn.parameters())
    def train(self):
        self.vae_dyn.train()
        for batch_idx, (cur_img_obs, goal_img_obs, action, _) in enumerate(self.train_loader):
            self.vae_dyn_optim.zero_grad()
            cur_img_obs = cur_img_obs.to(self.device).type("torch.cuda.FloatTensor")
            goal_img_obs = goal_img_obs.to(self.device).type("torch.cuda.FloatTensor")
            action = action.to(self.device)
            # feed cur_img_obs and goal_img_obs into encoder
            with torch.no_grad():
                z_cur, _, _ = self.vae.forward(cur_img_obs, encode_only=True)
                z_goal, _, _ = self.vae.forward(goal_img_obs, encode_only=True)
            out = self.vae_dyn.forward(z_cur, action)
            mse = nn.MSELoss()
            loss = mse(out, z_goal)
            wandb.log({"train_loss": loss})
            loss.backward()
            if batch_idx == 250:
                break
            self.vae_dyn_optim.step() 
    def eval(self):
        self.vae_dyn.eval()
        val_loss = 0
        stopping_amnt = 100
        for batch_idx, (cur_img_obs, goal_img_obs, action, _) in enumerate(self.val_loader):
            cur_img_obs = cur_img_obs.to(self.device).type("torch.cuda.FloatTensor")
            goal_img_obs = goal_img_obs.to(self.device).type("torch.cuda.FloatTensor")
            action = action.to(self.device)
            with torch.no_grad():
                z_cur, _, _ = self.vae.forward(cur_img_obs, encode_only=True)
                z_goal, _, _ = self.vae.forward(goal_img_obs, encode_only=True)
                out = self.vae_dyn.forward(z_cur, action)
                mse = nn.MSELoss()
                val_loss += mse(out, z_goal)
            if batch_idx == stopping_amnt:
                break
        wandb.log({"End of Epoch Val Loss": val_loss / stopping_amnt})

def main():
    args = parse_args()
    run_title = args.out_weights.split("/")[-1].split(".")[0]
    wandb.init(project="vae_dyn-training-fluid-manip", name=run_title)
    trainer = Trainer(args.in_vae_weights, args.in_trajs_file, args.ac_dim)
    for _ in range(15):
      trainer.train()
      torch.save(trainer.vae_dyn.state_dict(), args.out_weights)
      trainer.eval()
      
main()
