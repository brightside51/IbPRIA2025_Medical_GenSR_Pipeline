
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
from utils import *

from dataset_128 import *
from model_128_2 import *

from ssim import *

BATCH_SIZE = 1
gpu = True
workers = 4


path_slurm = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"

trainset = Dataset_LIDC(path_slurm)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                          shuffle=True,num_workers=workers)


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

####################################### CONFIGURATION ##############################################################



G = Generator(mode='test', latent_dim=1024, num_class=0).cuda()






#-----------------------
#Load Pre-trained model
#-----------------------

#------------Trained Model of LIDC Dataset---------------------
ckpt_path = '/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/HA-GAN/scripts/128/first_try_80k/checkpoint/HA_GAN_run1/G_iter80000.pth'
ckpt = torch.load(ckpt_path, map_location='cuda')
ckpt['model'] = trim_state_dict_name(ckpt['model'])
G.load_state_dict(ckpt['model'])

train_loader = torch.utils.data.DataLoader(trainset,batch_size = 2, shuffle=True, num_workers=workers)
sum_ssim = 0
for k in range(1):
    for i,dat in enumerate(train_loader):
        if len(dat)!=2:
            break
        img1 = dat[0]
        img2 = dat[1]

        msssim =msssim_3d(img1,img2)
        sum_ssim = sum_ssim+msssim
    print(f"MSSIM real: {sum_ssim/((k+1)*(i+1))}")

