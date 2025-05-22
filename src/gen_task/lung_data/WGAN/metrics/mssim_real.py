
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
from dataset_64 import *
from model_64 import *

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



#-----------------------
#Load Pre-trained model
#-----------------------

#------------Trained Model of LIDC Dataset---------------------

G.load_state_dict(torch.load('/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/WGAN/scripts/64x64/4th_try_new_generator/checkpoint_64_200k/G_2_iter130001.pth'))


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

