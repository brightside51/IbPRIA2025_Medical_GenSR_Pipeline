import os

import numpy as np
import torch
import time

from torch.utils.data import DataLoader, ConcatDataset
from data_pedro import get_training_set

from matplotlib import pyplot as plt
from matplotlib import image as mpimg


directory = '/nas-ctm01/homes/dcampas/dataset_orig/sortedT1W'

img_size = 64
num_frames = 64

print("Started to load dataset")
dataset_train = get_training_set(directory, img_size= img_size, num_frames = num_frames)
#data_inputs = next(iter(training_data_loader))
#print("Data inputs", data_inputs.shape, "\n", data_inputs)

print("Dataset created")






