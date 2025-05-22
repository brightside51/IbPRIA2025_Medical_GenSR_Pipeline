
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import nibabel as nib
from nilearn import plotting
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from PIL import Image

import sys
sys.path.append('../datasets')
sys.path.append('../models')
sys.path.append('../metrics')
from dataset_128 import *
from model_128 import *
from ssim import *

gpu = True
workers = 4
BATCH_SIZE = 1

path_slurm = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"

trainset = Dataset_LIDC(path_slurm)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                          shuffle=True,num_workers=workers)


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

####################################### CONFIGURATION ##############################################################

gpu = True
workers = 4


G = Generator(noise=1000).cuda()




G.load_state_dict(torch.load('/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/alpha-WGAN/scripts/checkpoint_128/second_try/G_128_iter200001.pth'))


################################ MS-SSIM Calculation ######################################################

max_value_MSSIM = 0
min_value_MSSIM = 10000000000
sum_ssim = 0
for k in range(2):
    for i,dat in enumerate(train_loader):
        noise_1 = Variable(torch.randn((1, 1000))).cuda()
        fake_image_1 = G(noise_1)
        img1 = dat[0]

        msssim = msssim_3d(img1,fake_image_1)
        sum_ssim = sum_ssim+msssim
        
        if msssim > max_value_MSSIM:  #Save the 2 images that resulted the max MSSIM
            max_value_MSSIM = msssim
            samples_dir = "max_MSSIM_real_128"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                # plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
            
            samples_dir = "max_MSSIM_fake_128"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                # plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
        if msssim < min_value_MSSIM:         #Save the 2 images that resulted the min MSSIM
            min_value_MSSIM= msssim
            samples_dir = "min_MSSIM_real_128"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                # plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
            samples_dir = "min_MSSIM_fake_128"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                # plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
    print(f"MSSIM dataset + generated: {sum_ssim/((k+1)*(i+1))}")
    print(f"Min value for MSSIM dataset + generated: {min_value_MSSIM}")
    print(f"Max value for MSSIM dataset + generated: {max_value_MSSIM}")