# Goal Conditioned Behavior Cloning (GCBC) Agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from dataloader import TrajectoryDataset
import itertools


class ConvModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, pad):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=kernel, 
            padding=pad, 
            stride=stride
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
        return nn.Sequential(
            conv_module_one,
            conv_module_two,
            conv_module_three
        )
        
    def build_mlp_layers_(self) -> nn.Sequential:
        flatten = nn.Flatten()
        l1 = MLPModule(64 * 84 * 84, 128)
        l2 = MLPModule(128, 128)
        l3 = MLPModule(128, self.ac_dim) # output a mean vector corresponding to the number of action components
        return nn.Sequential(
            flatten,
            l1,
            l2,
            l3
        )
    def __init__(self, ac_dim):
        super().__init__()
        self.ac_dim = ac_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logstd = nn.Parameter(torch.zeros((ac_dim), device=self.device))
        self.conv_layers = self.build_conv_layers_()
        self.mlp_layers = self.build_mlp_layers_()
        self.gcbc_optim = optim.Adam(
            itertools.chain(
            self.conv_layers.parameters(), 
            self.mlp_layers.parameters(), 
            [self.logstd]
            )
        )
    def forward(self, x) -> distributions.Distribution:
        means = self.mlp_layers(self.conv_layers(x)).to(self.device)
        return distributions.Normal(means, torch.exp(self.logstd))
    def update(self, obs_no, ac_na):
        dist = self.forward(obs_no)
        pred_ac_na = dist.sample()
        return pred_ac_na.to(self.device)
    
agent = GCBCAgent(5)
ex_input_obs_batch = torch.randn((5, 3, 256, 256))
ex_goal_obs_batch = torch.randn((5, 3, 256, 256))
ex_ac_batch = torch.randn((5, 5))
inpt = torch.cat((ex_input_obs_batch, ex_goal_obs_batch), 1)
print(agent.update(inpt, ex_ac_batch))