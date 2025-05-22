
import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from nilearn import plotting
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
#from dataset import *
from dataset_cross import *
#from models.alpha_WGAN_64 import *
from models.alpha_WGAN_128 import *
#from models.alpha_WGAN_64_26 import *
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from ipdb import set_trace
import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
# from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from nilearn import plotting



################################################### Configuration ############################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train and save models.")
    parser.add_argument("--total_iter", type=int, default=200001, help="Total number of iterations for training")
    parser.add_argument("--checkpoint_dir", type=str, default="checks_128_cross", help="Directory path to save models")
    parser.add_argument("--samples_dir", type=str, default="samples_train_128_cross", help="Directory path to save samples generated during training")
    parser.add_argument("--losses_ext", type=str, default="train_128_cross", help="Extension to distinct between losses")

    return parser.parse_args()

args = parse_args()
BATCH_SIZE=4
gpu = True
workers = 8
path_slurm_Duke = "/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1"
path_slurm_METABREST = "/nas-ctm01/datasets/private/METABREST/T1W_Breast"

#Use_Duke=False # Use METABREST dataset
Use_Duke =True
checkpoint_dir = args.checkpoint_dir
samples_dir = args.samples_dir
losses_ext = args.losses_ext


LAMBDA= 10
_eps = 1e-15

torch_seed = 0
torch.manual_seed(torch_seed)


# Check the number of available CUDA devices

#setting latent variable sizes
#latent_dim = 1000
#trainset = Dataset_Duke(path_slurm_METABREST)
#train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                          #shuffle=True,num_workers=workers)

#trainset = Dataset_Duke(path_slurm_Duke)
    #train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                          #shuffle=True,num_workers=workers)

latent_dim = 1000
trainset = Dataset_Duke()
train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                          shuffle=True,num_workers=workers)
    
def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images
        
            
            
G = Generator(noise = latent_dim)
CD = Code_Discriminator(code_size = latent_dim ,num_units = 4096)
D = Discriminator(is_dis=True)
#E = Discriminator(out_class = latent_dim,is_dis=False,img_size=64)
E = Discriminator(out_class = latent_dim,is_dis=False)                  #for 128x128x128

#G.load_state_dict(torch.load('_saves_128_cross/G_64_iter40001.pth'))
#CD.load_state_dict(torch.load('_saves_128_cross/CD_64_iter40001.pth'))
#D.load_state_dict(torch.load('_saves_128_cross/D_64_iter40001.pth'))
#E.load_state_dict(torch.load('_saves_128_cross/E_64_iter40001.pth'))

G.cuda()
D.cuda()
CD.cuda()
E.cuda()

g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
e_optimizer = optim.Adam(E.parameters(), lr = 0.0002)
cd_optimizer = optim.Adam(CD.parameters(), lr = 0.0002)


def calc_gradient_penalty(model, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size()==x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size).uniform_()
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+_eps).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty



################################ Trainning ###########################################################

torch.autograd.set_detect_anomaly(True)
#remove Variable usage
# real_y = torch.ones((BATCH_SIZE, 1)).cuda()#async=True))
# fake_y = torch.zeros((BATCH_SIZE, 1)).cuda()#async=True))

criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

# load the highest savepoints of all models
iteration = 0

g_iter = 1
d_iter = 1
cd_iter =1
TOTAL_ITER = args.total_iter
gen_load = inf_train_gen(train_loader)


D_loss_list = []
En_G_loss_list = []
C_loss_list = []

for iteration in range(TOTAL_ITER):
    for p in D.parameters():  
        p.requires_grad = False
    for p in CD.parameters():  
        p.requires_grad = False
    for p in E.parameters():  
        p.requires_grad = True
    for p in G.parameters():  
        p.requires_grad = True

    ###############################################
    # Train Encoder - Generator 
    ###############################################
     ###############################################
    # Train Encoder - Generator 
    ###############################################
    for iters in range(g_iter):
        G.zero_grad()
        E.zero_grad()
        real_images = gen_load.__next__()
        _batch_size = real_images.size(0)
        real_images = Variable(real_images,volatile=True).cuda()
        z_rand = Variable(torch.randn((_batch_size,latent_dim)),volatile=True).cuda()
        z_hat = E(real_images).view(_batch_size,-1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)
        c_loss = -CD(z_hat).mean()

        d_real_loss = D(x_hat).mean()
        d_fake_loss = D(x_rand).mean()
        d_loss = -d_fake_loss-d_real_loss
        l1_loss =10* criterion_l1(x_hat,real_images)
        loss1 = l1_loss + c_loss + d_loss
        
        if iters<g_iter-1:
            loss1.backward()
        else:
            loss1.backward(retain_graph=True, inputs=list(G.parameters()))
        e_optimizer.step() 
        g_optimizer.step()
        g_optimizer.step()
    ###############################################
    # Train D
    ###############################################
    for p in D.parameters():  
        p.requires_grad = True
    for p in CD.parameters():  
        p.requires_grad = False
    for p in E.parameters():  
        p.requires_grad = False
    for p in G.parameters():  
        p.requires_grad = False

    for iters in range(d_iter):
        d_optimizer.zero_grad()
        real_images = gen_load.__next__()
        _batch_size = real_images.size(0)
        z_rand = Variable(torch.randn((_batch_size,latent_dim)),volatile=True).cuda()
        real_images = Variable(real_images,volatile=True).cuda()
        z_hat = E(real_images).view(_batch_size,-1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)
        x_loss2 = -2*D(real_images).mean()+D(x_hat).mean()+D(x_rand).mean()
        gradient_penalty_r = calc_gradient_penalty(D,real_images.data, x_rand.data)
        gradient_penalty_h = calc_gradient_penalty(D,real_images.data, x_hat.data)

        loss2 = x_loss2+gradient_penalty_r+gradient_penalty_h
        loss2.backward(retain_graph=True, inputs=list(D.parameters()))
        d_optimizer.step()

    ###############################################
    # Train CD
    ###############################################
    for p in D.parameters():  
        p.requires_grad = False
    for p in CD.parameters():  
        p.requires_grad = True
    for p in E.parameters():  
        p.requires_grad = False
    for p in G.parameters():  
        p.requires_grad = False

    for iters in range(cd_iter):
        cd_optimizer.zero_grad()
        z_rand = Variable(torch.randn((_batch_size,latent_dim)),volatile=True).cuda()
        gradient_penalty_cd = calc_gradient_penalty(CD,z_hat.data, z_rand.data)
        loss3 = -CD(z_rand).mean() - c_loss + gradient_penalty_cd
        loss3.backward(retain_graph=True)
        cd_optimizer.step()

    ###############################################
    # Visualization
    ###############################################
    
    D_loss_list.append(loss2.item())
    En_G_loss_list.append(loss1.item())
    C_loss_list.append(loss3.item())
        
        
    if iteration % 5000 == 0:
        
        print('[{}/{}]'.format(iteration,TOTAL_ITER),
              'D: {:<8.3f}'.format(loss2.item()), 
              'En_Ge: {:<8.3f}'.format(loss1.item()),
              'Code: {:<8.3f}'.format(loss3.item()),
              )
        
        # Choose the index of the slice you want to visualize
        slice_index = 30
        if not os.path.exists(samples_dir):
        # If it doesn't exist, create the directory
            os.makedirs(samples_dir)
        
        feat = np.squeeze((0.5*real_images[0]+0.5).data.cpu().numpy())
        #print(len(feat))
        #feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plt.imsave(f'{samples_dir}/x_real{iteration}.png', feat[slice_index], cmap='gray')
        
        feat = np.squeeze((0.5*x_hat[0]+0.5).data.cpu().numpy())
        #feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plt.imsave(f'{samples_dir}/x_hat{iteration}.png', feat[slice_index], cmap='gray')


        feat = np.squeeze((0.5*x_rand[0]+0.5).data.cpu().numpy())
        #feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plt.imsave(f'{samples_dir}/x_rand{iteration}.png', feat[slice_index], cmap='gray')

        # Plot loss2
        plt.figure()
        plt.plot(D_loss_list, label='D_loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Discriminator loss Over Iterations')
        plt.legend()
        plt.savefig(f'D_loss_plot{losses_ext}.png')  # Save as PNG
        plt.close()

        # Plot loss1
        plt.figure()
        plt.plot(En_G_loss_list, label='En_G_loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Encoded Generator Loss Over Iterations')
        plt.legend()
        plt.savefig(f'En_G_loss_plot{losses_ext}.png')  # Save as PNG
        plt.close()

        # Plot loss3
        plt.figure()
        plt.plot(C_loss_list, label='C_loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Code Loss Over Iterations')
        plt.legend()
        plt.savefig(f'C_loss_plot{losses_ext}.png')  # Save as PNG
        plt.close()
        
        
    ###############################################
    # Model Save
    ###############################################
    
    directory_path = f"{checkpoint_dir}"

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(directory_path)
        if (iteration)%50000 ==0:
            torch.save(G.state_dict(),f'{directory_path}/G_64_iter'+str(iteration+1)+'.pth')
            torch.save(D.state_dict(),f'{directory_path}/D_64_iter'+str(iteration+1)+'.pth')
            torch.save(E.state_dict(),f'{directory_path}/E_64_iter'+str(iteration+1)+'.pth')
            torch.save(CD.state_dict(),f'{directory_path}/CD_64_iter'+str(iteration+1)+'.pth')
            print("Saved model")
    else:
        if (iteration)%50000 ==0:
            torch.save(G.state_dict(),f'{directory_path}/G_64_iter'+str(iteration+1)+'.pth')
            torch.save(D.state_dict(),f'{directory_path}/D_64_iter'+str(iteration+1)+'.pth')
            torch.save(E.state_dict(),f'{directory_path}/E_64_iter'+str(iteration+1)+'.pth')
            torch.save(CD.state_dict(),f'{directory_path}/CD_64_iter'+str(iteration+1)+'.pth')
            print("Saved model")

    
    iteration += 1
