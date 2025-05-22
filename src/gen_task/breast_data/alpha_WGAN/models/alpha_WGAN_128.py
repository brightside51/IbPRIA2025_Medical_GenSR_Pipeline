import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

#***********************************************
#Encoder and Discriminator has same architecture
#************************************************
class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1,is_dis =True):
        super(Discriminator, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class 
        
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
    
class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3
            
        return output

class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.noise = noise
        
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*16, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(_c*16)
        
        self.tp_conv2 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv3 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv4 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv5 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(_c)
        
        self.tp_conv6 = nn.Conv3d(_c, _c//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm3d(_c//2)
        
        self.tp_conv7 = nn.Conv3d(_c//2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):
        noise = noise.view(-1, self.noise, 1, 1, 1)
        h1 = self.relu(self.bn1(self.tp_conv1(noise)))
        
        h2 = F.interpolate(h1, scale_factor=2)
        h2 = self.relu(self.bn2(self.tp_conv2(h2)))
     
        h3 = F.interpolate(h2, scale_factor=2)
        h3 = self.relu(self.bn3(self.tp_conv3(h3)))

        h4 = F.interpolate(h3, scale_factor=2)
        h4 = self.relu(self.bn4(self.tp_conv4(h4)))

        h5 = F.interpolate(h4, scale_factor=2)
        h5 = self.relu(self.bn5(self.tp_conv5(h5)))
        
        h6 = F.interpolate(h5, scale_factor=2)
        h6 = self.relu(self.bn6(self.tp_conv6(h6)))

        h7 = F.interpolate(h6, scale_factor=2)
        h7 = self.tp_conv7(h7)

        h = torch.tanh(h7)

        return h