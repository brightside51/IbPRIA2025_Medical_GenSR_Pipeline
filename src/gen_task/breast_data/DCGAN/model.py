import torch
import torch.nn as nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, latent_dim, channels=1, slices=60):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.slices = slices
        self.output_channels = channels
        self.kernel_increase = (3, 4, 4)
        self.stride_increase = (1, 2, 2)
        self.padding_increase = (0,1,1)
        self.kernel_double = (4, 4, 4)
        self.stride_double = (2, 2, 2)
        self.padding_double = (1,1,1)
        
        self.decoder = nn.Sequential( 
            
            nn.ConvTranspose3d(latent_dim, 1024, self.kernel_double, self.stride_double, self.padding_double, bias=False),
            nn.BatchNorm3d(1024),
            nn.ReLU(True),

            nn.ConvTranspose3d(1024, 512, self.kernel_increase, self.stride_increase,self.padding_increase , bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, self.kernel_double, self.stride_double, self.padding_double, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            # 
            nn.ConvTranspose3d(256, 128, self.kernel_double, self.stride_double, self.padding_double, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(128, 64, self.kernel_double, self.stride_double, self.padding_double, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, self.output_channels, self.kernel_double, self.stride_double, self.padding_double, bias=False),
            #nn.ConvTranspose3d(self.output_channels, self.output_channels, self.kernel_increase, self.stride_increase, self.padding_increase, bias=False),

            nn.Tanh()
        
        )


       

    def forward(self, input):
        #latent = self.encoder(input)
        output = self.decoder(input)
        return output
    

class Discriminator(nn.Module):
    def __init__(self, channels=1, slices=60):
        super(Discriminator, self).__init__()
        self.slices = slices
        self.output_channels = channels
        self.kernel_decrease = (2, 4, 4)
        self.stride_decrease = (1, 2, 2)
        self.padding_decrease = (0,1,1)
        self.kernel_half = (4, 4, 4)
        self.stride_half = (2, 2, 2)
        self.padding_half = (1,1,1)
        self.base_output_channels = 64
        self.main = nn.Sequential(
            
            nn.Conv3d(channels, self.base_output_channels, self.kernel_half, self.stride_half, self.padding_half, bias=False),
            nn.BatchNorm3d(self.base_output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(self.base_output_channels, self.base_output_channels*2, self.kernel_half, self.stride_half, self.padding_half, bias=False),
            nn.BatchNorm3d(self.base_output_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(self.base_output_channels*2, self.base_output_channels*(2**2), self.kernel_half, self.stride_half, self.padding_half, bias=False),
            nn.BatchNorm3d(self.base_output_channels*(2**2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(self.base_output_channels*4, self.base_output_channels*(2**3), self.kernel_half, self.stride_half, self.padding_half, bias=False),
            nn.BatchNorm3d(self.base_output_channels*(2**3)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(self.base_output_channels*(2**3), self.base_output_channels*(2**4), self.kernel_half, self.stride_half, self.padding_half, bias=False),
            nn.BatchNorm3d(self.base_output_channels*(2**4)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(self.base_output_channels*(2**4) ,1, self.kernel_half, self.stride_half, self.padding_half), # 1x1x1 output
            #nn.Conv3d(1 ,1, self.kernel_half, self.stride_half, self.padding_half), # 1x1x1 output
            # nn.Conv2d(1 * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        # squeeze the output to 1D
        output = output.view(-1, 1).squeeze(1)
        return output