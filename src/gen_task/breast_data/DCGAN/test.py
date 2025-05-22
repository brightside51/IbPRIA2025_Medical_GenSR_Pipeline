import os
import sys
sys.path.append(os.getcwd())
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import argparse

import numpy as np
import torch
import time
from model import Generator, Discriminator
from main import dataset_train, num_frames
#from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, ConcatDataset
#from metric import calculate_fid_score

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader

from scipy.linalg import sqrtm
from tqdm import tqdm

import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image

from metric import  calculate_fid_score


def main():
    # --------------------Initialize the generator and discriminator----------------------------
    latent_dim = 100
    generator = Generator(latent_dim, channels=1, slices=64)  
    discriminator = Discriminator(channels=1, slices=64)  
    batch_size = 20     

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))  
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #---------------------------load dos modelos------------------------------------
    checkpoint = torch.load("/nas-ctm01/homes/dcampas/DCGAN/checkpoints_64.pth")       
    generator.load_state_dict(checkpoint['generator_state_dict'])           
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    print("Models loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    training_data_loader = DataLoader(dataset_train, num_workers=0, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        print("Passing models to cuda")
        generator.cuda()
        discriminator.cuda()
        print("Models are cuda'd")

    #---------------------Start testing-----------------------------
    for i, real_data in enumerate(training_data_loader):
        print(f"Batch_{i} ") 
        noise = torch.randn(real_data.size(0), latent_dim, 1, 1, 1, device=device)
        generated_volumes = generator.decoder(noise)
        
        combined_lists = zip(generated_volumes, real_data)
        p=0
        # For each patient generated we have a FID score
        for fake, real in combined_lists:
            save_test_images(fake, i, p)
            p+=1
            #print(fake.shape)
            fake_2 = fake.permute(1, 0, 2,3)
            real_2 = real.permute(1, 0, 2,3)
            rgb_batch_fake = fake_2.repeat(1, 3, 1, 1)
            rgb_batch_real = real_2.repeat(1, 3, 1, 1)
            
            #print(rgb_batch_real.shape)
            # Calculate FID score
            fid_score = calculate_fid_score(rgb_batch_real, rgb_batch_fake)
            print("FID score:", fid_score)
    #-------------Calculate FID score------------------------------------------      
    #gen = []
    #reale = []
    #generated = "/nas-ctm01/homes/dcampas/DCGAN/generated_test/batch_0"
    #real = "/nas-ctm01/homes/dcampas/DCGAN/sortedT1W_png64/ID10"

    #for path in os.listdir(generated):
        #patient_path = os.path.join(generated, path)
        #for image in os.listdir(patient_path):
            #image_path = os.path.join(patient_path, image)
            #gen.append(image_path)
            

    #for image_2 in os.listdir(real):
        #image_path_2 = os.path.join(real, image_2)
        #reale.append(image_path_2)



    # Set the desired image size (e.g., 128x128)
    #image_size = 64

    # Load images
    #real_images = load_images(reale, image_size)
    #fake_images = load_images(gen, image_size)

    # Calculate FID score
    #fid_score = calculate_fid_score(real_images, fake_images)
    #print("FID score:", fid_score)
           

            



def save_test_images(data, i,p):
    
    try:
        os.makedirs(f'/nas-ctm01/homes/dcampas/DCGAN/generated_test/batch_{i}', exist_ok = True)
    
    except OSError as error:
        print("error")
        
    out = os.path.join(f'/nas-ctm01/homes/dcampas/DCGAN/generated_test/batch_{i}', f'patient_{p}')  
    
    try:
        os.makedirs(out, exist_ok = True)
    
    except OSError as error:
        print("error")
        
    data = data.permute(1, 0, 2, 3)
    k=0
    for mri in data:
        #if not os.path.exists(output_folder):
           # os.makedirs(output_folder)
        for image in mri:
            image_array = image.cpu().detach().numpy() * 255.0
            # Save the image as a file in the folder
            image_path = os.path.join(out, f'image_{k}.png')  # Specify the image file path
            cv2.imwrite(image_path, image_array)
            k+=1
    
    print("Images saved")


if __name__ == "__main__":
    main()
    print("Ran")
    exit(0)