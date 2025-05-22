from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import nibabel as nib
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import sys
sys.path.append('../datasets')
sys.path.append('../models')
from dataset_128 import *
from model_128 import *
from ssim import *

gpu = True
workers = 4
BATCH_SIZE = 1

path_slurm = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"

trainset = Dataset_LIDC(path_slurm)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)

# Model Initialization
G = Generator(noise=1000).cuda()
G.load_state_dict(torch.load('/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/WGAN/scripts/128x128/G_128_190K.pth'))

#################################### Maximum-Mean Discrepancy Score #################################
max_value_MMD = 0
min_value_MMD = float('inf')
train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=workers)
for p in G.parameters():
    p.requires_grad = False

meanarr = []
for s in range(100):
    distmean = 0.0
    for i, (y) in enumerate(train_loader):
        y = Variable(y).cuda()
        noise = Variable(torch.randn((y.size(0), 1000)).cuda())
        x = G(noise)

        B = y.size(0)
        x = x.view(x.size(0), x.size(2) * x.size(3) * x.size(4))
        y = y.view(y.size(0), y.size(2) * y.size(3) * y.size(4))

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        beta = (1. / (B * B))
        gamma = (2. / (B * B))

        Dist = beta * (torch.sum(xx) + torch.sum(yy)) - gamma * torch.sum(zz)
        distmean += Dist 

    value_distmean = distmean / (i + 1)

    if value_distmean > max_value_MMD:
        max_value_MMD = value_distmean

    if value_distmean < min_value_MMD:
        min_value_MMD = value_distmean

    meanarr.append(value_distmean)

meanarr = np.array([item.cpu().detach().numpy() for item in meanarr])
print('Total_mean:', np.mean(meanarr), 'STD:', np.std(meanarr))
print("Min for MMD:", min_value_MMD)
print("Max for MMD:", max_value_MMD)
