
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

#from dataset import *
from dataset_cross import *
#from models.alpha_WGAN_64_26 import Generator
from models.Model_HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder
#from models.alpha_WGAN_128 import *

from metrics.pytorch_ssim import *
from utils import *

latent_dim = 1024
gpu = True
workers = 8
BATCH_SIZE = 1

path_slurm_Duke = "/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1"
path_slurm_METABREST = "/nas-ctm01/datasets/private/METABREST/T1W_Breast"


#'flair' or 't2' or 't1ce'
#trainset = Dataset_Duke(path_slurm, min_slices=True) #Test min slices
#trainset = Dataset_Duke(path_slurm_Duke)
trainset = Dataset_Duke()

train_loader = torch.utils.data.DataLoader(trainset,batch_size = BATCH_SIZE, shuffle=True, num_workers=workers)




def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

####################################### CONFIGURATION ##############################################################

gpu = True
workers = 4


G = Generator(mode='test', latent_dim=1024, num_class=0).cuda()




#-----------------------
#Load Pre-trained model
#-----------------------

#------------Trained Model of Duke Dataset---------------------

#G.load_state_dict(torch.load('checkpoint_200k_more64/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('checkpoint_o_200k_64/G_64_iter400001.pth'))
#G.load_state_dict(torch.load('checkpoint_Duke_200k_64_min/G_64_iter200001.pth'))
#G.load_state_dict(torch.load('COPD_HA_GAN_pretrained/G_iter80000.pth'))

ckpt_path = '/nas-ctm01/homes/dcampas/diogocampasmsc/HA_GAN/checkpoint/HA_GAN_run2/G_iter80000.pth'
#ckpt_path = '/nas-ctm01/homes/dcampas/diogocampasmsc/HA_GAN/checkpoint/HA_GAN_run5_128_META/G_iter80000.pth'
#ckpt_path = '/nas-ctm01/homes/dcampas/diogocampasmsc/HA_GAN/checkpoint/HA_GAN_run6_128_crossbalance/G_iter80000.pth'
#ckpt_path = '/nas-ctm01/homes/dcampas/diogocampasmsc/HA_GAN/checkpoint/HA_GAN_run_Duke_128/G_iter80000.pth'
ckpt = torch.load(ckpt_path, map_location='cuda')
ckpt['model'] = trim_state_dict_name(ckpt['model'])
G.load_state_dict(ckpt['model'])

######################################### Generate and save 50 samples ##########################################################################

k=0
for k in range (50):
    noise_1 = torch.randn((1, latent_dim)).cuda()
    fake_image_1 = G(noise_1,0)
    featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
    samples_dir_1 = "Generated_50"
    if not os.path.exists(samples_dir_1):
        os.makedirs(samples_dir_1)
    
    c=0
    for img in featmask:
        samples_dir_2 = f"Sample_{k}"
        if not os.path.exists(f"{samples_dir_1}/{samples_dir_2}"):
            os.makedirs(f"{samples_dir_1}/{samples_dir_2}")
            
        plt.imsave(f'{samples_dir_1}/{samples_dir_2}/slice{c}.png', img, cmap='gray')
        c=c+1
    k=k+1

gen_load = inf_train_gen(train_loader)

l=0
for l in range (20):
    image = gen_load.__next__()
    featmask = np.squeeze(0.5*image[0]+0.5).data.cpu().numpy()
    samples_dir_1 = "Real_50"
    if not os.path.exists(samples_dir_1):
        os.makedirs(samples_dir_1)
    
    c=0
    for img in featmask:
        samples_dir_2 = f"Sample_{l}"
        if not os.path.exists(f"{samples_dir_1}/{samples_dir_2}"):
            os.makedirs(f"{samples_dir_1}/{samples_dir_2}")
            
        plt.imsave(f'{samples_dir_1}/{samples_dir_2}/slice{c}.png', img, cmap='gray')
        c=c+1
    
       
################################ MS-SSIM Calculation ######################################################

max_value_MSSIM = 0
min_value_MSSIM = 10000000000
sum_ssim = 0
for k in range(5):
    for i,dat in enumerate(train_loader):
        noise_1 = torch.randn((1, latent_dim)).cuda()
        fake_image_1 = G(noise_1,0)
        img1 = dat[0]

        msssim = msssim_3d(img1,fake_image_1)
        sum_ssim = sum_ssim+msssim
        
        if msssim > max_value_MSSIM:  #Save the 2 images that resulted the max MSSIM
            max_value_MSSIM = msssim
            samples_dir = f"max_MSSIM_real_{k}_{i}"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
            
            samples_dir = f"max_MSSIM_fake_{k}_{i}"
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
            samples_dir = "min_MSSIM_real"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*img1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
                
            samples_dir = "min_MSSIM_fake"
            if not os.path.exists(samples_dir):
                # If it doesn't exist, create the directory
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5*fake_image_1[0]+0.5).data.cpu().numpy()
            c=0
            for img in featmask:
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')
                c=c+1
        
    print(f"MSSIM dataset + generated: {sum_ssim/((k+1)*(i+1))}")
    print(f"Min value for MSSIM dataset + generated: {min_value_MSSIM}")
    print(f"Max value for MSSIM dataset + generated: {max_value_MSSIM}")

N = 100  # Number of fake images
sum_ssim = 0
batch_size = 10  # Choose an appropriate batch size
max_value_MSSIM = 0
min_value_MSSIM = float('inf')

mssim_scores = []  # List to store all MSSIM scores

for i in range(0, N, batch_size):
    batch_ssim = 0
    for j in range(batch_size):
        noise_1 = torch.randn((1, latent_dim)).cuda()
        noise_2 = torch.randn((1, latent_dim)).cuda()
        fake_image_1 = G(noise_1, 0)
        fake_image_2 = G(noise_2, 0)

        if torch.cuda.is_available():
            img1 = fake_image_1.cuda()
            img2 = fake_image_2.cuda()

        score = msssim_3d(img1, img2)
        mssim_scores.append(score.item())  # Collect MSSIM scores

        if score > max_value_MSSIM:  # Save the 2 images that resulted in the max MSSIM
            max_value_MSSIM = score
            samples_dir = "max_MSSIM_img1"
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5 * img1[0] + 0.5).data.cpu().numpy()
            for c, img in enumerate(featmask):
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')

            samples_dir = "max_MSSIM_img2"
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5 * img2[0] + 0.5).data.cpu().numpy()
            for c, img in enumerate(featmask):
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')

        if score < min_value_MSSIM:  # Save the 2 images that resulted in the min MSSIM
            min_value_MSSIM = score
            samples_dir = "min_MSSIM_img1"
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5 * img1[0] + 0.5).data.cpu().numpy()
            for c, img in enumerate(featmask):
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')

            samples_dir = "min_MSSIM_img2"
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
            featmask = np.squeeze(0.5 * img2[0] + 0.5).data.cpu().numpy()
            for c, img in enumerate(featmask):
                plt.imsave(f'{samples_dir}/slice{c}.png', img, cmap='gray')

        batch_ssim += score.item()  # Accumulate SSIM score for the batch

    sum_ssim += batch_ssim / batch_size  # Average SSIM score per batch

# Calculate the mean MSSIM score
mean_mssim = sum_ssim / (N / batch_size)

# Calculate the standard deviation of MSSIM scores
std_mssim = np.std(mssim_scores)

print(f"MSSIM Fake: {mean_mssim}")
print(f"Min value for MSSIM Fake: {min_value_MSSIM}")
print(f"Max value for MSSIM Fake: {max_value_MSSIM}")
print(f"Standard deviation for MSSIM Fake: {std_mssim}")

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
train_loader = torch.utils.data.DataLoader(trainset,batch_size = 1, shuffle=True, num_workers=4)
for p in G.parameters():
    p.requires_grad = False


meanarr = []
for s in range(1):
    distmean = 0.0
    for i,(y) in enumerate(train_loader):
        y = Variable(y).cuda()
        #print(y.size())
        z_rand_1 = torch.randn((y.size(0), 1024)).cuda()
        #rint("SEPARATE")
        x = G(z_rand_1,0).cuda()
        #print(x.size())
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
    #print(value_distmean)
    
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
    
    





