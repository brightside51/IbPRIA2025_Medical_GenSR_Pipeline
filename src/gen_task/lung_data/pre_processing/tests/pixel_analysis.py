import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom

# Path to the directory containing the DICOM files
input_directory = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/LIDC_LungSegmentation/LIDC-IDRI-0011/1-081.npy"


slice_data = np.load(input_directory)
    
# Calculate range, max, min, and mean pixel values
 # Get the initial size of the image
initial_size = slice_data.shape
pixel_range = np.ptp(slice_data)
pixel_max = np.max(slice_data)
pixel_min = np.min(slice_data)
pixel_mean = np.mean(slice_data)

# Print pixel value statistics
print(f"File: {input_directory}")
print(f"Initial Image Size: {initial_size}")
print(f"Pixel Range: {pixel_range}")
print(f"Max Pixel Value: {pixel_max}")
print(f"Min Pixel Value: {pixel_min}")
print(f"Mean Pixel Value: {pixel_mean}")