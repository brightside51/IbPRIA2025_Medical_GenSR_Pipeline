import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        n_class = 1
        
        self.conv1 = nn.Conv3d(1, channel//16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//8)
        self.conv3 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//4)
        self.conv4 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel//2)
        self.conv5 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(channel)
        self.conv6 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)
        output = h6
        

        return output
    



class Generator(nn.Module):
    def __init__(self, noise:int=1000, channel:int=64):
        super(Generator, self).__init__()
        _c = channel
        
        self.noise = noise
        self.fc = nn.Linear(1000, 512*8*8*8)  # Adjusted output size for 8x8x8 feature maps
        self.bn1 = nn.BatchNorm3d(512)
        
        self.tp_conv2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(256)
        
        self.tp_conv3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.tp_conv4 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(64)
        
        self.tp_conv5 = nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        
    def forward(self, noise):
        noise = noise.view(-1, 1000)
        h = self.fc(noise)
        h = h.view(-1, 512, 8, 8, 8)  # Reshape to 512 channels of size 8x8x8
        h = F.relu(self.bn1(h))

        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        h = self.tp_conv5(h)

        h = torch.tanh(h)

        return h
