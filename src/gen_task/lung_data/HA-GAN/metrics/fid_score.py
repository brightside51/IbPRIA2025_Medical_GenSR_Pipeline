#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image


import sys
sys.path.append('../datasets')
sys.path.append('../models')
from dataset_128 import *
from model_128 import *

from utils import *

workers = 8
BATCH_SIZE = 1


try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')

latent_dim = 1024
args = parser.parse_args()

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images


G = Generator(mode='test', latent_dim=1024, num_class=0).cuda()
ckpt_path = '/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/HA-GAN/scripts/128/first_try_80k/checkpoint/HA_GAN_run1/G_iter80000.pth'
#ckpt_path = '/nas-ctm01/homes/dcampas/diogocampasmsc/HA_GAN/checkpoint/HA_GAN_run6_128_crossbalance/G_iter80000.pth'
#ckpt_path = '/nas-ctm01/homes/dcampas/diogocampasmsc/HA_GAN/checkpoint/HA_GAN_run_Duke_128/G_iter80000.pth'
ckpt = torch.load(ckpt_path, map_location='cuda')
ckpt['model'] = trim_state_dict_name(ckpt['model'])
G.load_state_dict(ckpt['model'])

def get_activations_fake(images, model, batch_size=1, dims=2048,
                    cuda=True, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    
    pred_arr = np.empty((128, dims))
    c=0
    
    #images = images.cuda()
    images_rgb = images.repeat(1, 3, 1, 1, 1)
    images_rgb = images_rgb.squeeze(0)
    #print(images_rgb.size())
    images_final = images_rgb.permute(1, 0, 2, 3)
    
    if cuda:
        images_final = images_final.cuda()

    pred = model(images_final)[0]
    start=c
    end=c+128

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def get_activations_real(images, model, batch_size=1, dims=2048,
                    cuda=True, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((128, dims))
    c=0
    
    #images=images.cuda()
    print(images.size())
    images_rgb = images.repeat(1, 3, 1, 1, 1)
    #print(images_rgb.size())
    images_rgb = images_rgb.squeeze(0)
    #print(images_rgb.size())
    images_final = images_rgb.permute(1, 0, 2, 3)
    #print(images_final.size())
    if cuda:
        images_final = images_final.cuda()

    #print(images.size())
    pred = model(images_final)[0]
    start=c
    end=c+128

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
    

    if verbose:
        print(' done')

    return pred_arr

def calculate_fid_given_paths(batch_size, cuda=True, dims=2048):
    """Calculates the FID of real and fake images"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    
    model.cuda()

    path_slurm = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"

    trainset = Dataset_LIDC(path_slurm)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,
                                            shuffle=True,num_workers=workers)
    
    sum_fid = 0
    count=1
    for i, images in enumerate(train_loader):
        
        if count >=400:
            return sum_fid / count
        
        else:
            with torch.no_grad():  # Disable gradient calculation
                noise_1 = torch.randn((1, latent_dim)).cuda()
                fake_images = G(noise_1,0).cuda()
                images=images.cuda()
                m1, s1 = _compute_statistics_of_real(images, model, batch_size, dims, cuda)
                m2, s2 = _compute_statistics_of_fake(fake_images, model, batch_size, dims, cuda)
                fid_value = calculate_frechet_distance(m1, s1, m2, s2)
                print(f"FID_{i}: {fid_value}")

                sum_fid += fid_value

                # Explicitly release memory
                del images, fake_images, m1, s1, m2, s2
                torch.cuda.empty_cache()

    return sum_fid / (trainset.__len__())


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics_real(images, model, batch_size=1,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_real(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_fake(images, model, batch_size=1,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_fake(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_real(images,model, batch_size, dims, cuda):
    
    m, s = calculate_activation_statistics_real(images,model, batch_size,
                                            dims, cuda)

    return m, s

def _compute_statistics_of_fake(images, model, batch_size, dims, cuda):
    
    m, s = calculate_activation_statistics_fake(images, model, batch_size,
                                            dims, cuda)

    return m, s




if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_value = calculate_fid_given_paths(args.batch_size,
                                          args.gpu != '',
                                          args.dims)
    print('FID: ', fid_value)