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
    def __init__(self, root_path, image_size=64, num_frames=64, min_slices=False) :
        super(Dataset_LIDC, self).__init__()
        self.root_path = root_path # folder with mri
        self.image_dirnames = [f.path for f in os.scandir(root_path) if f.is_dir() and f.name != exclude_name and f.name != exclude_name_2] #Gather names of directories with images        self.num_frames = self.get_minimum_images(root_path)
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
        
    # def order_dicom_files(self, dicom_directory):
    #     # Step 1: Iterate through each DICOM file
    #     json_directory = 'path_to_json_directory'
    #     sorted_dicom_directory = 'path_to_sorted_dicom_directory'

    #     dicom_files = os.listdir(dicom_directory)

    #     # Step 4: Create a mapping between DICOM file paths and their corresponding attribute values
    #     attribute_values = {}

    #     for dicom_file in dicom_files:
    #         dicom_path = os.path.join(dicom_directory, dicom_file)
            
    #         # Step 2: Extract the corresponding JSON file
    #         json_file_name = os.path.splitext(dicom_file)[0] + '.json'
    #         json_path = os.path.join(json_directory, json_file_name)
            
    #         # Step 3: Read the attribute value from the JSON file
    #         with open(json_path, 'r') as json_file:
    #             json_data = json.load(json_file)
    #             attribute_value = json_data['your_attribute_name']  # Replace 'your_attribute_name' with the actual attribute name
            
    #         attribute_values[dicom_path] = attribute_value

    #     # Step 5: Sort the DICOM file paths based on the extracted attribute values
    #     sorted_files = sorted(attribute_values.items(), key=lambda x: x[1])

    #     # Step 6: Reorganize the directory according to the sorted file paths
    #     for idx, (file_path, _) in enumerate(sorted_files):
    #         new_file_name = f"{idx + 1}_{os.path.basename(file_path)}"
    #         sorted_file_path = os.path.join(sorted_dicom_directory, new_file_name)
    #         os.rename(file_path, sorted_file_path)

    
    
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
        # im = self.normalize_dicom_pixels(im)
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
    
    def normalize_dicom_pixels(dicom_data, maxHU=400., minHU=-1000.):
        # Normalize function to map HU values to range [0, 1]
        normalized_pixel_data = (dicom_data.pixel_array - minHU) / (maxHU - minHU)
        normalized_pixel_data[normalized_pixel_data < 0] = 0
        normalized_pixel_data[normalized_pixel_data > 1] = 1
        return normalized_pixel_data
    
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


        tensors_tuple =  tuple(tensors_filt)
        stack = torch.stack(tensors_tuple, dim = 1)
        stack = stack*2-1
        #print(torch.stack(tensors_tuple, dim = 1))
        #return torch.stack(tensors_tuple, dim = 1)
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

        
#trainset = Dataset_Duke(path_local)      

#number = trainset.get_minimum_images(path_local)

#print(number)
 
