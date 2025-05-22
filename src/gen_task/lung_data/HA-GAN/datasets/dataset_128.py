import os
import numpy as np
import pydicom
import cv2

from PIL import Image
import torch

from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms as T, utils

import copy
from natsort import natsorted

from torchvision.datasets.folder import is_image_file

from torch.utils.data import dataloader

exclude_name = "metadata"
exclude_name_2 = "LICENSE"


class Dataset_LIDC(Dataset):
    def __init__(self, root_path, image_size=128, num_frames=64, min_slices=False) :
        super(Dataset_LIDC, self).__init__()
        self.root_path = root_path # folder with mri
        self.image_dirnames = [f.path for f in os.scandir(root_path) if f.is_dir() and f.name != exclude_name and f.name != exclude_name_2] #Gather names of directories with images        
        #self.num_frames = self.get_minimum_images(root_path)
        self.min_slices = min_slices
        
        if self.min_slices:
            self.num_frames = self.get_minimum_images(root_path)
        
        else:
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

            dicom_files = [f for f in os.listdir(root) if f.endswith('.dcm')]
            if(len(dicom_files) < 10):
                continue
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext.lower() in ['', '.dcm']:
                    file_path = os.path.join(root, filename)
                    if self.passes_filter(root, filename):
                        names.append(file_path)
        return names

    def passes_filter(self, directory, filename):
        path_components = directory.split('/')
        desired_part = path_components[8]

        new_suffix = ".npy"
        modified_filename = filename[:-4] + new_suffix

        # Split the filename into base and extension
        base, ext = os.path.splitext(modified_filename)

        if '1' not in base.split('-')[0]:
            return False  # Skip files without '1' in the prefix
        # Split the base into prefix and number part
        prefix, number = base.rsplit('-', 1)
        # Pad the number part with zeros to make it three digits
        padded_number = number.zfill(3)
        # Concatenate the prefix and padded number with the extension
        new_filename = f"{prefix}-{padded_number}{ext}"

        slice_mask = np.load("/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/LIDC_LungSegmentation/" + desired_part + "/" + new_filename)
        if(cv2.countNonZero(slice_mask) != 0):
            return True
        return False


    def convert_dcm_png(self, name):
        im = pydicom.dcmread(name, force=True)
        im_orientation = str(im[0x00200037].value)
        im_position = str(im[0x00200013].value)

        im = im.pixel_array.astype(float)

        rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
        final_image = np.uint8(rescaled_image) # integers pixels

        final_image = Image.fromarray(final_image)
        final_image = final_image.resize((64,64))

        do_v_flip = False

        if(im_orientation == "[-1, 0, 0, 0, -1, 0]"):
            do_v_flip = True

        return final_image, im_position, do_v_flip
    
   #  (channels, frame, height, width) tensor
    def __getitem__(self, sampleIdx):

        do_h_flip = 1
        if torch.rand(1) < 0.5:
            do_h_flip = 0

        # print("SampleIdx")
        # print(sampleIdx)

        dir_path = self.image_dirnames[sampleIdx]
        names = self.get_names(os.path.join(self.root_path,dir_path))
        sorted_files = natsorted(names)

        # print(dir_path)
        num_images = len(sorted_files)

        

        if(num_images >= 64):
            if torch.rand(1) < 0.5:
                slides = [sorted_files[i] for i in range(num_images) if i % 2 != 0]
            else:
                slides = [sorted_files[i] for i in range(num_images) if i % 2 == 0]
        else:
            slides = sorted_files

        
        tensors = [None]*764

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
        duplicated_list = []
        #duplicated_list_2 = []
        #print(num_mris)
        if self.min_slices:
            num_samples = self.num_frames
            tensors_filt = self.ordered_sampling(tensors_filt, num_samples)  # Uncomment this to use sampling
            
        else:
            # If less than max_size images, duplicate last until reaching max
            if(num_mris < max_size):
                num_duplicates = max_size - num_mris
                last_element = tensors_filt[-1]
                for i in range(0,num_duplicates): tensors_filt.append(last_element)
                
            
            # If more than max_size images, get only middle section
            elif(num_mris > max_size):
                start = (num_mris - max_size) // 2
                end = start + max_size
                tensors_filt = tensors_filt[start:end]
        
        for item in tensors_filt:
            duplicated_list.extend([item, item])
        
        #for item in duplicated_list:
            #duplicated_list_2.extend([item,item])
        
        
        tensors_tuple =  tuple(duplicated_list)
        stack = torch.stack(tensors_tuple, dim = 1)
        stack = stack*2-1
        #print(torch.stack(tensors_tuple, dim = 1))
        #return torch.stack(tensors_tuple, dim = 1)
        #print(stack.size())
        return stack

    
    def __len__(self):
        return len(self.image_dirnames)
    
    def count_images_in_directory(self, directory):
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', 'dcm']  # Add more extensions if needed
        image_count = 0
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isdir(filepath):
                # If it's a subdirectory, recursively count images
                image_count += self.count_images_in_directory(filepath)
            elif any(filename.lower().endswith(ext) for ext in image_extensions):
                # If it's an image file, increment the count
                image_count += 1
        return image_count

    def get_minimum_images(self, root_directory):
        
        c = 0
        min_images_per_patient = float('inf')
        for patient_dir in os.listdir(root_directory):
            if patient_dir != exclude_name and patient_dir!= exclude_name_2: 
                patient_dir_path = os.path.join(root_directory, patient_dir)
                if os.path.isdir(patient_dir_path):
                    # If it's a directory (patient directory), count images
                    #print(patient_dir_path)
                    image_count = self.count_images_in_directory(patient_dir_path)
                    #print(image_count)
                    if image_count < min_images_per_patient:
                        min_images_per_patient = image_count
                        #print(image_count)
                    
                    #if image_count < 64:
                        #c= c+1
                        
                    
        #print(min_images_per_patient)
        #print(c)
        return min_images_per_patient
    
    def ordered_sampling(self, images, num_samples):
        
        step = len(images) // num_samples  # Adjust step size according to the number of samples needed
        remainder = len(images) % num_samples  # Calculate the remainder
        sampled_images = [images[i * step + min(i, remainder)] for i in range(num_samples)]
        return sampled_images
        

 
