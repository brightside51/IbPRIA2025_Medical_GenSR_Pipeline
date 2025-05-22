import os
import numpy as np
import pydicom

from PIL import Image
import torch

from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms as T, utils

import copy
from natsort import natsorted

from torchvision.datasets.folder import is_image_file


class DatasetFromDICOM(Dataset):
    def __init__(self, root_path, image_size=64, num_frames=30) :
        super(DatasetFromDICOM, self).__init__()
        self.root_path = root_path # folder with mri
        self.image_dirnames = [f.path for f in os.scandir(root_path) if f.is_dir()] #Gather names of directories with images
        self.num_frames = num_frames
        
        #These are some data augmentation techniques to make our dataset more diverse
        self.transform_h = T.Compose([
            T.RandomHorizontalFlip(p=1),
            T.Resize((image_size,image_size)),
            T.ToTensor()
        ])

        self.transform_h_v = T.Compose([
            T.RandomHorizontalFlip(p=1),
            T.RandomVerticalFlip(p=1),
            T.Resize((image_size,image_size)),
            T.ToTensor()
        ])

        self.transform = T.Compose([
            T.Resize((image_size,image_size)),
            T.ToTensor()
        ])

        self.transform_v = T.Compose([
            T.RandomVerticalFlip(p=1),
            T.Resize((image_size,image_size)),
            T.ToTensor()
        ])


        #self.image_dirnames = self.image_dirnames[0:2]    #DEBUGGING ONLY
        
    def get_names(self, path):
        names = []
        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext in ['','.dcm']:
                    names.append(os.path.join(root, filename))
        
        return names

    def convert_dcm_png(self, name):
        im = pydicom.dcmread(name,  force=True)
        im_orientation = str(im[0x00200037].value)
        im_position = str(im[0x00200013].value)

        im = im.pixel_array.astype(float)

        rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
        final_image = np.uint8(rescaled_image) # integers pixels

        final_image = Image.fromarray(final_image)
        final_image = final_image.resize((128,128))

        do_v_flip = False

        if(im_orientation == "[-1, 0, 0, 0, -1, 0]"):
            do_v_flip = True

        return final_image, im_position, do_v_flip
    
    #  (channels, frame, height, width) tensor
    def __getitem__(self, sampleIdx):

        do_h_flip = 1
        if torch.rand(1) < 0.5:
            do_h_flip = 0
        
        dir_path = self.image_dirnames[sampleIdx]
        names = self.get_names(os.path.join(self.root_path,dir_path))
        sorted_files = natsorted(names)

        num_images = len(sorted_files)

        if(num_images >= 60):
            if torch.rand(1) < 0.5:
                slides = [sorted_files[i] for i in range(num_images) if i % 2 != 0]
            else:
                slides = [sorted_files[i] for i in range(num_images) if i % 2 == 0]
        else:
            slides = sorted_files
        
        tensors = [None]*100

        for img in slides:
            # mri_image = Image.open(os.path.join(self.root_path, dir_path, img))
            mri_image, position, do_v_flip = self.convert_dcm_png(img)

            if(do_h_flip and do_v_flip):
                tensors[int(position)-1] = self.transform_h_v(mri_image)
            elif(do_h_flip and not do_v_flip):
                tensors[int(position)-1] = self.transform_h(mri_image)
            elif (not do_h_flip and do_v_flip):
                tensors[int(position)-1] = self.transform_v(mri_image)
            else:
                tensors[int(position)-1] = self.transform(mri_image)
        
        tensors_filt = list(filter(lambda item: item is not None, tensors))

        max_size = self.num_frames
        num_mris = len(tensors_filt)
    
        # If less than 60 images, duplicate last until reaching max
        if(num_mris < max_size):
            num_duplicates = max_size - num_mris
            last_element = tensors_filt[-1]
            for i in range(0,num_duplicates): tensors_filt.append(last_element)

        # If more than 60 images, get only middle section
        elif(num_mris > max_size):
            start = (num_mris - max_size) // 2
            end = start + max_size
            tensors_filt = tensors_filt[start:end]

        tensors_tuple =  tuple(tensors_filt)
        return torch.stack(tensors_tuple, dim = 1)

    
    def __len__(self):
        return len(self.image_dirnames)



class DatasetFromFolder(Dataset):
    def __init__(self, root_path, image_size=64):
        super(DatasetFromFolder, self).__init__()
        self.root_path = root_path # folder with mri
        self.image_dirnames = [x for x in os.listdir(self.root_path)] #Gather names of directories with images

        self.transform_flip = T.Compose([
            T.RandomHorizontalFlip(p=1),
            T.Resize((image_size,image_size)),
            T.ToTensor()
        ])

        self.transform = T.Compose([
            T.Resize((image_size,image_size)),
            T.ToTensor()
        ])

        #self.image_dirnames = self.image_dirnames[0:2]    #DEBUGGING ONLY
        
    
    #  (channels, frame, height, width) tensor
    def __getitem__(self, sampleIdx):

        do_h_flip = 1
        if torch.rand(1) < 0.5:
            do_h_flip = 0
        
        dir_path = self.image_dirnames[sampleIdx]
        tensors = []

        sorted_files = natsorted(os.listdir(os.path.join(self.root_path, dir_path)))

        if torch.rand(1) < 0.5:
            slides = [sorted_files[i] for i in range(len(sorted_files)) if i % 2 != 0]
        else:
            slides = [sorted_files[i] for i in range(len(sorted_files)) if i % 2 == 0]
        
        for img in slides:
            mri_image = Image.open(os.path.join(self.root_path, dir_path, img))

            if(do_h_flip):
                tensors.append(self.transform_h(mri_image))
            else:
                tensors.append(self.transform(mri_image))
        
        max_size = 30
        num_mris = len(tensors)
    
        # If less than 60 images, duplicate last until reaching max
        if(num_mris < max_size):
            num_duplicates = max_size - num_mris
            last_element = tensors[-1]
            for i in range(0,num_duplicates): tensors.append(last_element)

        # If more than 60 images, get only middle section
        elif(num_mris > max_size):
            start = (num_mris - max_size) // 2
            end = start + max_size
            tensors = tensors[start:end]

        tensors_tuple =  tuple(tensors)
        return torch.stack(tensors_tuple, dim = 1)

    
    def __len__(self):
        return len(self.image_dirnames)

#Create training dataset
def get_training_set(root_dir, img_size, num_frames):
    train_dir = os.path.join(root_dir)
    return DatasetFromDICOM(train_dir, img_size, num_frames)

# #Create validation dataset
# def get_val_set(root_dir):
#     val_dir = os.path.join(root_dir, "val")
#     return DatasetFromFolder(val_dir)
# 
# #Create test dataset
# def get_test_set(root_dir):
#     test_dir = os.path.join(root_dir, "test")
#     return DatasetFromFolder(test_dir)
