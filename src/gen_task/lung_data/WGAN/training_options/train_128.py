import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
import sys
sys.path.append('../datasets')
sys.path.append('../models')
from dataset_128 import *
from model_128 import *
from PIL import Image
import argparse
import matplotlib.pyplot as plt

######################################### CONFIGURATION ########################################


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save models.")
    parser.add_argument("--total_iter", type=int, default=400001, help="Total number of iterations for training")
    parser.add_argument("--latent_dim", type=int, default=1000, help="Size of latent variable")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_o_400k_64", help="Directory path to save models")
    parser.add_argument("--samples_dir", type=str, default="samples_64_400k", help="Directory path to save samples generated during training")
    parser.add_argument("--losses_ext", type=str, default="_64_o", help="Extension to distinct between losses")
    parser.add_argument("--slices_size", type=int, default=64, help="Size of each slice")


    return parser.parse_args()


args = parse_args()
BATCH_SIZE=4
max_epoch = 100
lr = 0.0001
gpu = True
workers = 4

path_slurm = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"



checkpoint_dir = args.checkpoint_dir
samples_dir = args.samples_dir
losses_ext = args.losses_ext
size_slices = args.slices_size

LAMBDA= 10
#setting latent variable sizes
latent_dim = 1000



# Check if we have a GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


trainset = Dataset_LIDC(path_slurm)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                          shuffle=True,num_workers=workers)

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images
            
            
D = Discriminator()
G = Generator(noise = latent_dim)

#G.load_state_dict(torch.load('checkpoint_o_200k_64/G_64_iter200001.pth'))
#D.load_state_dict(torch.load('checkpoint_o_200k_64/D_64_iter200001.pth'))

G.cuda()
D.cuda()

g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)


G.load_state_dict(torch.load('/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/WGAN/scripts/128x128/checkpoint_o_200k_128/G_64_iter90001.pth'))
D.load_state_dict(torch.load('/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/WGAN/scripts/128x128/checkpoint_o_200k_128/D_64_iter90001.pth'))


def calc_gradient_penalty(netD, real_data, fake_data):    
    alpha = torch.rand(real_data.size(0),1,1,1,1)
    alpha = alpha.expand(real_data.size())
    
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


##################################### TRAINING ###################################################


real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
loss_f = nn.BCELoss()

d_real_losses = list()
d_fake_losses = list()
d_losses = list()
g_losses = list()
divergences = list()


GENERATOR_TRAINING_RATIO = 5  # Number of times to train G for each D training step

TOTAL_ITER = args.total_iter
gen_load = inf_train_gen(train_loader)
for iteration in range(TOTAL_ITER):
    ###############################################
    # Train D 
    ###############################################
    for p in D.parameters():  
        p.requires_grad = True 

    real_images = gen_load.__next__()
    D.zero_grad()
    real_images = Variable(real_images).to(device=device)

    _batch_size = real_images.size(0)

    y_real_pred = D(real_images)

    d_real_loss = y_real_pred.mean()
    
    noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1, 1))).to(device=device)
    fake_images = G(noise)
    y_fake_pred = D(fake_images.detach())

    d_fake_loss = y_fake_pred.mean()
    gradient_penalty = calc_gradient_penalty(D, real_images.data, fake_images.data)
 
    d_loss = -d_real_loss + d_fake_loss + gradient_penalty
    d_loss.backward()
    Wasserstein_D = d_real_loss - d_fake_loss

    d_optimizer.step()

    ###############################################
    # Train G 
    ###############################################
    for p in D.parameters():
        p.requires_grad = False

    for _ in range(GENERATOR_TRAINING_RATIO):
        G.zero_grad()
        noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1, 1))).to(device=device)
        fake_image = G(noise)
        y_fake_g = D(fake_image)

        g_loss = -y_fake_g.mean()

        g_loss.backward()
        g_optimizer.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    ###############################################
    # Visualization
    ###############################################
    if iteration%10000 == 0:
        d_real_losses.append(d_real_loss.item())
        d_fake_losses.append(d_fake_loss.item())
        

        print('[{}/{}]'.format(iteration,TOTAL_ITER),
              'D: {:<8.3}'.format(d_loss.item()), 
              'D_real: {:<8.3}'.format(d_real_loss.item()),
              'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
              'G: {:<8.3}'.format(g_loss.item()))

        
        # Choose the index of the slice you want to visualize
        slice_index = 13
        if not os.path.exists(samples_dir):
        # If it doesn't exist, create the directory
            os.makedirs(samples_dir)
        
        feat = np.squeeze((0.5*fake_image[0]+0.5).data.cpu().numpy())
        #print(len(feat))
        #feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plt.imsave(f'{samples_dir}/x_rand{iteration}.png', feat[12], cmap='gray')
        
        
    ###############################################
    # Model Save
    ###############################################
    
    directory_path = f"{checkpoint_dir}"

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(directory_path)
        if (iteration)%10000 ==0:
            torch.save(G.state_dict(),f'{directory_path}/G_64_iter'+str(iteration+1)+'.pth')
            torch.save(D.state_dict(),f'{directory_path}/D_64_iter'+str(iteration+1)+'.pth')
            print("Saved model")
    else:
        if (iteration)%10000 ==0:
            torch.save(G.state_dict(),f'{directory_path}/G_64_iter'+str(iteration+1)+'.pth')
            torch.save(D.state_dict(),f'{directory_path}/D_64_iter'+str(iteration+1)+'.pth')
            print("Saved model")

            # Plot loss2
            plt.figure()
            plt.plot(d_losses, label='D_loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Discriminator loss Over Iterations')
            plt.legend()
            plt.savefig(f'D_loss_plot{losses_ext}.png')  # Save as PNG
            plt.close()

            # Plot loss1
            plt.figure()
            plt.plot(g_losses, label='G_loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Generator Loss Over Iterations')
            plt.legend()
            plt.savefig(f'G_loss_plot{losses_ext}.png')  # Save as PNG
            plt.close()




