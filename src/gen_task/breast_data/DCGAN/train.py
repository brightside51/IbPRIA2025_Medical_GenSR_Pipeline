# flake8: noqa
print("____train____")
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

import os

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
   
    # Initialize the generator and discriminator
    latent_dim = 100
    generator = Generator(latent_dim, channels=1, slices=64)  #criação do gerador
    #generator = Generator()  #criação do gerador
    discriminator = Discriminator(channels=1, slices=64)       #criação do discriminador
    #discriminator = Discriminator() 
    args = parser.parse_args()
    
    if args.train:
        print('Retraining model')
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        checkpoint = torch.load("checkpoints_64.pth")
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    else:
        print('Training new model')
        # Define loss function and optimizer
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    
    criterion = nn.BCELoss()  #binary-cross entropy loss
    # Training loop
    num_epochs = 1000   # nº de epochs
    batch_size = 20     # nº de pacientes para treino
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")
    # Assuming you have a dataset of 3D volumes or point clouds
    training_data_loader = DataLoader(dataset_train, num_workers=0, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        print("Passing models to cuda")
        generator.cuda()
        discriminator.cuda()
        print("Models are cuda'd")
    
    #train_also_discriminator= True
    
    for epoch in range(num_epochs):
        for i, real_data in enumerate(training_data_loader):
            
            #k += batch_size
    #----------------Treino do discriminador---------------------------
    
            discriminator.zero_grad()
            
            real_data = real_data.to(device)
            real_labels = torch.full((real_data.size(0),), 1.0, device=device)
            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            
            real_loss.backward()
                
            
            noise = torch.randn(real_data.size(0), latent_dim, 1, 1, 1, device=device)
            noise = noise.to(device)
            fake_data = generator(noise)
            
                        
            fake_output = discriminator(fake_data.detach())
            fake_labels = torch.full((fake_output.size(0),), 0.0, device=device)
            fake_loss = criterion(fake_output, fake_labels)
            fake_loss.backward() 
            
            d_loss = real_loss + fake_loss
            
            
            
            
    #-------------------- Treino do gerador-------------------------------------------------------------
            
            generator.zero_grad()
            
            fake_output = discriminator(fake_data)
            generator_real_labels = torch.full(fake_output.shape, 1.0, device=device) # Ver se isto é correto
            g_loss = criterion(fake_output, generator_real_labels)
            g_loss.backward()
            
            
            optimizer_G.step()
            
            if d_loss>g_loss:
                optimizer_D.step()

            
    #-----------------------Prints e saves---------------------------------------------------------------
            #train_also_discriminator = d_loss > g_loss # variavel para ver se treino o gerador ou nao de forma a ultrapassar problemas que advem de treinar GANs
            print(epoch)
            if epoch % 100 == 0:
                
                print(
                        f"Epoch [{epoch}/{num_epochs}], "
                        f"Batch [{i}/{len(training_data_loader)}], "
                        f"Discriminator Loss: {d_loss.item():.4f}, "
                        f"Generator Loss: {g_loss.item():.4f}"
                    )
                
                torch.save({
                    'epoch':epoch,
                    'generator_state_dict':generator.state_dict(),
                    'discriminator_state_dict':discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict':optimizer_D.state_dict(),   
                }, 'checkpoints_64.pth')
                
    #----------------------------------Save all patients generated -------------------------------------------
            if (epoch % 100 == 0):
                p=0
                for patients in fake_data:
                    save_images(patients,epoch,i, p)
                    p+=1
                


def save_images(data,epoch, i,p):
    
    if i == 0 or i==1:
        return 0
    
    try:
        os.makedirs(f'/nas-ctm01/homes/dcampas/DCGAN/images_output/epoch_{epoch}', exist_ok = True)
    
    except OSError as error:
        print("error")
        
    out = os.path.join(f'/nas-ctm01/homes/dcampas/DCGAN/images_output/epoch_{epoch}', f'patient_{p}')  
    
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
            image_path = os.path.join(out, f'image_{k}_e{epoch}.png')  # Specify the image file path
            cv2.imwrite(image_path, image_array)
            k+=1
    
    print("Images saved")


if __name__ == "__main__":
    main()
    print("Ran")
    exit(0)