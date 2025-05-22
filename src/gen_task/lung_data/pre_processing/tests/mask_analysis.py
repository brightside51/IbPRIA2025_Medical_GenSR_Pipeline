import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the directory containing the .npy files
input_directory = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/LIDC_LungSegmentation/LIDC-IDRI-0011"

# Path to the directory where you want to save the images
output_directory = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/test_images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get a list of .npy files in the input directory
npy_files = [file for file in os.listdir(input_directory) if file.endswith('.npy')]

# Loop through each .npy file
for npy_file in npy_files:
    # Load the .npy file
    slice_data = np.load(os.path.join(input_directory, npy_file))
    
    # Normalize pixel values to the range [0, 255]
    slice_data = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))) * 255
    
    # Convert to uint8 for saving as image
    slice_data = slice_data.astype(np.uint8)
    
    # Plot and save the slice as an image
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.savefig(os.path.join(output_directory, os.path.splitext(npy_file)[0] + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
