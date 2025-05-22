import itertools
import numpy as np
import torch
import os
import json
import argparse
from torch import nn, optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import nibabel as nib
from nilearn import plotting
from utils import trim_state_dict_name, inf_train_gen
from dataset_cross import *
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch HA-GAN Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    return config

def train(config):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    
    trainset = Dataset_Duke()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], drop_last=True,
                                               shuffle=False, num_workers=config['workers'])
    gen_load = inf_train_gen(train_loader)
    
    if config['img_size'] == 256:
        from models.Model_HA_GAN_256 import Discriminator, Generator, Encoder, Sub_Encoder
    if config['img_size'] == 128:
        from models.Model_HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder
    
    G = Generator(mode='train', latent_dim=config['latent_dim'], num_class=config['num_class']).cuda()
    D = Discriminator(num_class=config['num_class']).cuda()
    E = Encoder().cuda()
    Sub_E = Sub_Encoder(latent_dim=config['latent_dim']).cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(0.0,0.999), eps=1e-8)
    d_optimizer = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(0.0,0.999), eps=1e-8)
    e_optimizer = optim.Adam(E.parameters(), lr=config['lr_e'], betas=(0.0,0.999), eps=1e-8)
    sub_e_optimizer = optim.Adam(Sub_E.parameters(), lr=config['lr_e'], betas=(0.0,0.999), eps=1e-8)

    if config['continue_iter'] != 0:
        ckpt_path = f"./checkpoint/{config['exp_name']}/G_iter{config['continue_iter']}.pth"
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
        g_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = f"./checkpoint/{config['exp_name']}/D_iter{config['continue_iter']}.pth"
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = f"./checkpoint/{config['exp_name']}/E_iter{config['continue_iter']}.pth"
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        E.load_state_dict(ckpt['model'])
        e_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = f"./checkpoint/{config['exp_name']}/Sub_E_iter{config['continue_iter']}.pth"
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        Sub_E.load_state_dict(ckpt['model'])
        sub_e_optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        print(f"Ckpt {config['exp_name']} {config['continue_iter']} loaded.")

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    E = nn.DataParallel(E)
    Sub_E = nn.DataParallel(Sub_E)

    G.train()
    D.train()
    E.train()
    Sub_E.train()

    real_y = torch.ones((config['batch_size'], 1)).cuda()
    fake_y = torch.zeros((config['batch_size'], 1)).cuda()

    loss_f = nn.BCEWithLogitsLoss()
    loss_mse = nn.L1Loss()

    fake_labels = torch.zeros((config['batch_size'], 1)).cuda()
    real_labels = torch.ones((config['batch_size'], 1)).cuda()

    summary_writer = SummaryWriter(f"./checkpoint/{config['exp_name']}")

    with open(os.path.join(f"./checkpoint/{config['exp_name']}", 'configs.json'), 'w') as f:
        json.dump(config, f, indent=2)

    for p in D.parameters(): p.requires_grad = False
    for p in G.parameters(): p.requires_grad = False
    for p in E.parameters(): p.requires_grad = False
    for p in Sub_E.parameters(): p.requires_grad = False
    
    best_d_loss = float('inf')
    best_iteration = 0
    
    for iteration in range(config['continue_iter'], config['num_iter']):
        for p in D.parameters(): p.requires_grad = True
        for p in Sub_E.parameters(): p.requires_grad = False

        real_images = gen_load.__next__()
        D.zero_grad()
        real_images = real_images.float().cuda()
        real_images_small = F.interpolate(real_images, scale_factor=0.25)
        crop_idx = np.random.randint(0, config['img_size'] * 7 / 8 + 1)
        real_images_crop = real_images[:, :, crop_idx:crop_idx + config['img_size'] // 8, :, :]

        if config['num_class'] == 0:
            y_real_pred = D(real_images_crop, real_images_small, crop_idx)
            d_real_loss = loss_f(y_real_pred, real_labels)

            noise = torch.randn((config['batch_size'], config['latent_dim'])).cuda()
            fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=None)
            y_fake_pred = D(fake_images, fake_images_small, crop_idx)
        else:
            class_label_onehot = F.one_hot(class_label, num_classes=config['num_class'])
            class_label = class_label.long().cuda()
            class_label_onehot = class_label_onehot.float().cuda()
            y_real_pred, y_real_class = D(real_images_crop, real_images_small, crop_idx)
            d_real_loss = loss_f(y_real_pred, real_labels) + F.cross_entropy(y_real_class, class_label)

            noise = torch.randn((config['batch_size'], config['latent_dim'])).cuda()
            fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=class_label_onehot)
            y_fake_pred, y_fake_class = D(fake_images, fake_images_small, crop_idx)

        d_fake_loss = loss_f(y_fake_pred, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        for p in D.parameters(): p.requires_grad = False
        for p in G.parameters(): p.requires_grad = True

        for iters in range(config['g_iter']):
            G.zero_grad()
            noise = torch.randn((config['batch_size'], config['latent_dim'])).cuda()
            if config['num_class'] == 0:
                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=None)
                y_fake_g = D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels)
            else:
                fake_images, fake_images_small = G(noise, crop_idx=crop_idx, class_label=class_label_onehot)
                y_fake_g, y_fake_g_class = D(fake_images, fake_images_small, crop_idx)
                g_loss = loss_f(y_fake_g, real_labels) + config['lambda_class'] * F.cross_entropy(y_fake_g_class, class_label)

            g_loss.backward()
            g_optimizer.step()

        for p in E.parameters(): p.requires_grad = True
        for p in G.parameters(): p.requires_grad = False
        E.zero_grad()
        z_hat = E(real_images_crop)
        x_hat = G(z_hat, crop_idx=None)
        e_loss = loss_mse(real_images_crop, x_hat)
        e_loss.backward()
        e_optimizer.step()

        for p in E.parameters(): p.requires_grad = False
        for p in Sub_E.parameters(): p.requires_grad = True
        Sub_E.zero_grad()
        
        with torch.no_grad():
            z_hat_i_list = []
            # Process all sub-volume and concatenate
            for crop_idx_i in range(0,config['img_size'],config['img_size']//8):
                real_images_crop_i = real_images[:,:,crop_idx_i:crop_idx_i+config['img_size']//8,:,:]
                z_hat_i = E(real_images_crop_i)
                z_hat_i_list.append(z_hat_i)
            z_hat = torch.cat(z_hat_i_list, dim=2).detach()   
        sub_z_hat = Sub_E(z_hat)
        
        # Reconstruction
        if config['num_class'] == 0: # unconditional
            sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx)
        else: # conditional
            sub_x_hat_rec, sub_x_hat_rec_small = G(sub_z_hat, crop_idx=crop_idx, class_label=class_label_onehot)
        
        sub_e_loss = (loss_mse(sub_x_hat_rec,real_images_crop) + loss_mse(sub_x_hat_rec_small,real_images_small))/2.

        sub_e_loss.backward()
        sub_e_optimizer.step()

        # Logging
        if iteration%config['sample_iter'] == 0:
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('D_real', d_real_loss.item(), iteration)
            summary_writer.add_scalar('D_fake', d_fake_loss.item(), iteration)
            summary_writer.add_scalar('G_fake', g_loss.item(), iteration)
            summary_writer.add_scalar('E', e_loss.item(), iteration)
            summary_writer.add_scalar('Sub_E', sub_e_loss.item(), iteration)

        ###############################################
        # Visualization with Tensorboard
        ###############################################
        if iteration%200 == 0:
            print('[{}/{}]'.format(iteration,config['num_iter']),
                  'D_real: {:<8.3}'.format(d_real_loss.item()),
                  'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
                  'G_fake: {:<8.3}'.format(g_loss.item()),
                  'Sub_E: {:<8.3}'.format(sub_e_loss.item()),
                  'E: {:<8.3}'.format(e_loss.item()))
            
            save_dir = "image_outputs"
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            
            featmask = np.squeeze((0.5*real_images_crop[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(config['img_size']//2,config['img_size']//2,config['img_size']//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Real', fig, iteration, close=True)
            real_save_path = os.path.join(save_dir, "real_image.png")
            plotting.plot_img(featmask, title="REAL", cut_coords=(config['img_size']//2, config['img_size']//2, config['img_size']//16), draw_cross=False, cmap="gray")
            plt.savefig(real_save_path)
            plt.close()

            featmask = np.squeeze((0.5*sub_x_hat_rec[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REC",cut_coords=(config['img_size']//2,config['img_size']//2,config['img_size']//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Rec', fig, iteration, close=True)
            rec_save_path = os.path.join(save_dir, "reconstructed_image.png")
            plotting.plot_img(featmask, title="REC", cut_coords=(config['img_size']//2, config['img_size']//2, config['img_size']//16), draw_cross=False, cmap="gray")
            plt.savefig(rec_save_path)
            plt.close()
            
            
            featmask = np.squeeze((0.5*fake_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(config['img_size']//2,config['img_size']//2,config['img_size']//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Fake', fig, iteration, close=True)
            fake_save_path = os.path.join(save_dir, "fake_image.png")
            plotting.plot_img(featmask, title="FAKE", cut_coords=(config['img_size']//2, config['img_size']//2, config['img_size']//16), draw_cross=False, cmap="gray")
            plt.savefig(fake_save_path)
            plt.close()
            
        if iteration > 30000 and (iteration+1)%10000 == 0:
            torch.save({'model':G.state_dict(), 'optimizer':g_optimizer.state_dict()},'./checkpoint/'+config['exp_name']+'/G_iter'+str(iteration+1)+'.pth')
            torch.save({'model':D.state_dict(), 'optimizer':d_optimizer.state_dict()},'./checkpoint/'+config['exp_name']+'/D_iter'+str(iteration+1)+'.pth')
            torch.save({'model':E.state_dict(), 'optimizer':e_optimizer.state_dict()},'./checkpoint/'+config['exp_name']+'/E_iter'+str(iteration+1)+'.pth')
            torch.save({'model':Sub_E.state_dict(), 'optimizer':sub_e_optimizer.state_dict()},'./checkpoint/'+config['exp_name']+'/Sub_E_iter'+str(iteration+1)+'.pth')
        
        if d_loss.item() < best_d_loss:
            best_d_loss = d_loss.item()
            best_iteration = iteration

    summary_writer.close()
    return best_d_loss, best_iteration


def grid_search():
    hyperparameter_space = {
        'batch_size': [4, 8],
        'lr_g': [1e-4, 2e-4],
        'lr_d': [1e-4, 2e-4],
        'lr_e': [1e-4, 2e-4],
        'latent_dim': [1024, 1800],
        'num_class': [0],
        'img_size': [128],
        'g_iter': [1, 2],
        'num_iter': [80000,100000],
        'save_iter': [5000],
        'sample_iter': [1000],
        'exp_name': ['HA_GAN_optimal'],
        'continue_iter': [0],
        'workers': [8],
        'lambda_class': [1.0]
    }

    keys, values = zip(*hyperparameter_space.items())
    results = []
    for v in itertools.product(*values):
        config = dict(zip(keys, v))
        print(f"Running experiment with configuration: {config}")
        best_d_loss, best_iteration = train(config)
        results.append((config, best_d_loss, best_iteration))
    
    # Save results to a CSV file
    with open('grid_search_results.csv', 'w', newline='') as csvfile:
        fieldnames = list(hyperparameter_space.keys()) + ['best_d_loss', 'best_iteration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for config, best_d_loss, best_iteration in results:
            row = {**config, 'best_d_loss': best_d_loss, 'best_iteration': best_iteration}
            writer.writerow(row)

    # Identify the best configuration
    best_result = min(results, key=lambda x: x[1])
    print(f"Best configuration: {best_result[0]}")
    print(f"Best discriminator loss: {best_result[1]} at iteration {best_result[2]}")

if __name__ == "__main__":
    config = parse_args()
    grid_search()