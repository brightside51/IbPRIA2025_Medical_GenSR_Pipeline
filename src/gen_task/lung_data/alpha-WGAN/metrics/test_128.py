
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



#-----------------------
#Load Pre-trained model
#-----------------------

#------------Trained Model of Duke Dataset---------------------

#G.load_state_dict(torch.load('checkpoint_200k_more64/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('checkpoint_o_200k_64/G_64_iter400001.pth'))
#G.load_state_dict(torch.load('checkpoint_Duke_200k_64_min/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('checkpoint_Duke_200k_64/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('checkpoint_META_200k_64/G_64_iter100001.pth'))
#G.load_state_dict(torch.load('checkpoint_Duke_200k_128/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('checkpoint_cross_200k_128/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('checkpoint_cross_200k_64_mesmo/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('saves_64_cross/G_64_iter200001.pth'))
G.load_state_dict(torch.load('/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/alpha-WGAN/scripts/checkpoint_128/second_try/G_128_iter200001.pth'))

k=0
for k in range(50):
    noise_1 = torch.randn((1, 1000)).cuda()
    fake_image_1 = G(noise_1).cuda()
    featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
    c=0
    samples_dir_1 = "Generated_50"
    if not os.path.exists(samples_dir_1):
        os.makedirs(samples_dir_1)
        
    for img in featmask:
        samples_dir_2 = f"Sample_{k}"
        if not os.path.exists(f"{samples_dir_1}/{samples_dir_2}"):
            # If it doesn't exist, create the directory
            os.makedirs(f"{samples_dir_1}/{samples_dir_2}")
        plt.imsave(f'{samples_dir_1}/{samples_dir_2}/slice{c}.png', img, cmap='gray')
        c=c+1
    k=k+1
    
       
################################ MS-SSIM Calculation ######################################################

train_loader = torch.utils.data.DataLoader(trainset,batch_size = 1, shuffle=True, num_workers=workers) #Real
test_loader = inf_train_gen(train_loader)

max_value_MSSIM = 0
min_value_MSSIM = 10000000000
sum_ssim = 0
for k in range(1):
    for i,dat in enumerate(train_loader):
        noise_1 = Variable(torch.randn((1, 1000))).cuda()
        fake_image_1 = G(noise_1)
        img1 = dat[0]

        msssim = msssim_3d(img1,fake_image_1)
        sum_ssim = sum_ssim+msssim
        
        if msssim > max_value_MSSIM:  #Save the 2 images that resulted the max MSSIM
            max_value_MSSIM = msssim
            samples_dir = f"max_MSSIM_real_{i}_{k}"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
            
            samples_dir = f"max_MSSIM_fake_{i}_{k}"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
        if msssim < min_value_MSSIM:         #Save the 2 images that resulted the min MSSIM
            min_value_MSSIM= msssim
            samples_dir = f"min_MSSIM_real__{i}_{k}"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
            samples_dir = f"min_MSSIM_fake__{i}_{k}"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
    print(f"MSSIM dataset + generated: {sum_ssim/((k+1)*(i+1))}")


N=1000                                                                   #Fake
sum_ssim = 0
batch_size = 10  # Choose an appropriate batch size
max_value_MSSIM = 0
min_value_MSSIM = 10000000000

for i in range(0, N, batch_size):
    batch_ssim = 0
    for j in range(batch_size):
        noise_1 = Variable(torch.randn((1, 1000))).cuda()
        noise_2 = Variable(torch.randn((1, 1000))).cuda()
        fake_image_1 = G(noise_1)
        fake_image_2 = G(noise_2)

        if torch.cuda.is_available():
            img1 = fake_image_1.cuda()
            img2 = fake_image_2.cuda()
        
        score = msssim_3d(img1,img2)
        
        if score > max_value_MSSIM:  #Save the 2 images that resulted the max MSSIM
            max_value_MSSIM = score
            samples_dir = "max_MSSIM_img1"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
            
            samples_dir = "max_MSSIM_img2"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img2[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
        if score < min_value_MSSIM:         #Save the 2 images that resulted the min MSSIM
            min_value_MSSIM= score
            samples_dir = "min_MSSIM_img1"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
            samples_dir = "min_MSSIM_img2"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img2[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
        batch_ssim += score.item()  # Accumulate SSIM score for the batch

    sum_ssim += batch_ssim / batch_size  # Average SSIM score per batch

print(f"MSSIM Fake : {sum_ssim / (N / batch_size)}")
print(f"Min value for MSSIM Fake: {min_value_MSSIM}")
print(f"Max value for MSSIM Fake: {max_value_MSSIM}")

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

        
#################################### Maximum-Mean Discrepancy Score #################################
max_value_MMD = 0
min_value_MMD = 10000000000
train_loader = torch.utils.data.DataLoader(trainset,batch_size = 1, shuffle=True, num_workers=workers)
for p in G.parameters():
    p.requires_grad = False

meanarr = []
for s in range(100):
    distmean = 0.0
    for i,(y) in enumerate(train_loader):
        y = Variable(y).cuda()
        noise = Variable(torch.randn((y.size(0), 1000)).cuda())
        x = G(noise)

        B = y.size(0)
        x = x.view(x.size(0), x.size(2) * x.size(3)*x.size(4))
        y = y.view(y.size(0), y.size(2) * y.size(3)*y.size(4))

        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        beta = (1./(B*B))
        gamma = (2./(B*B)) 

        Dist = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
        distmean += Dist
    #print('Mean:'+str(distmean/(i+1)))
    value_distmean = distmean/(i+1)
    
    
    if value_distmean > max_value_MMD:  #Save the 2 images that resulted the max MSSIM
        max_value_MMD = value_distmean
        samples_dir = "max_MMD_img1"
        if not os.path.exists(samples_dir):
            # If it doesn't exist, create the directory
            os.makedirs(samples_dir)
        featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
        c=0
        for img in featmask:
            plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
            c=c+1
        
        samples_dir = "max_MMD_img2"
        if not os.path.exists(samples_dir):
            # If it doesn't exist, create the directory
            os.makedirs(samples_dir)
        featmask = np.squeeze(0.5*img2[0]+0.5).data.cpu().numpy()
        c=0
        for img in featmask:
            plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
            c=c+1
        
    if value_distmean < min_value_MMD:         #Save the 2 images that resulted the min MSSIM
        min_value_MMD= value_distmean
        samples_dir = "min_MMD_img1"
        if not os.path.exists(samples_dir):
            # If it doesn't exist, create the directory
            os.makedirs(samples_dir)
        featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
        c=0
        for img in featmask:
            plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
            c=c+1
            
        samples_dir = "min_MMD_img2"
        if not os.path.exists(samples_dir):
            # If it doesn't exist, create the directory
            os.makedirs(samples_dir)
        featmask = np.squeeze(0.5*img2[0]+0.5).data.cpu().numpy()
        c=0
        for img in featmask:
            plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
            c=c+1
    
    meanarr.append(value_distmean)
#meanarr = np.array(meanarr)
meanarr = np.array([item.cpu().detach().numpy() for item in meanarr])
print('Total_mean:'+str(np.mean(meanarr))+' STD:'+str(np.std(meanarr)))
print("Min for MMD: "+str(min_value_MMD))
print("Max for MMD: "+str(max_value_MMD))
    
    
