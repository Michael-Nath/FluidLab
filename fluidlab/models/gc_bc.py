# Goal Conditioned Behavior Cloning (GCBC) Agent

import torch
import torch.nn as nn
from dataloader import TrajectoryDataset


class ConvModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, pad):
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
        self.l1 = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.l1(x)))

class GCBCAgent(nn.Module):
    def build_conv_layers_(self) -> nn.Sequential:
        # We will take in a concatenation of the current scene obs and a goal configuration 
        conv_module_one = ConvModule(6, 32, 3, 1, 1)
        conv_module_two = ConvModule(32, 64, 3, 1, 0)
        conv_module_three = ConvModule(64, 128, 3, 1, 0)
        return nn.Sequential(
            ("conv_module_one", conv_module_one),
            ("conv_module_two", conv_module_two),
            ("conv_module_three", conv_module_three),
        )
        
    def build_mlp_layers_(self) -> nn.Sequential:
        l1 = MLPModule(-1, 128)
        l2 = MLPModule(128, 128)
        l3 = MLPModule(128, self.ac_dim) # output a mean vector corresponding to the number of action components
        return nn.Sequential(
            ("linear_one", l1),
            ("linear_two", l2),
            ("linear_three", l3)
        )
    def __init__(self, ac_dim):
        self.ac_dim = ac_dim
        self.conv_layers = self.build_conv_layers_()
        self.mlp_layers = self.build_mlp_layers_()
    def forward(self, x):
        return self.mlp_layers(self.conv_layers(x))