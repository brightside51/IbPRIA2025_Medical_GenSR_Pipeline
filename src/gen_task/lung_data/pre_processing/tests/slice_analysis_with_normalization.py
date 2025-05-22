import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the .npy files
directory_path = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images/LIDC-IDRI-1011"
output_path = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/npy_results_with_HU_normalization"

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# List all the files in the directory
file_names = os.listdir(directory_path)

# Filter only .npy files
npy_files = [file for file in file_names if file.endswith('.npy')]

# Normalize function to map HU values to range [0, 1]
def normalizePlanes(npzarray, maxHU = 400., minHU = -1000.):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1
    npzarray[npzarray < 0] = 0
    return npzarray.astype('float32')

# Iterate through each .npy file
for i, file_name in enumerate(npy_files):
    # Construct the full file path
    file_path = os.path.join(directory_path, file_name)

    # Load the .npy file
    slice_data = np.load(file_path)

    # Apply HU normalization
    slice_data_normalized = normalizePlanes(slice_data)

    # Display the image
    plt.imshow(slice_data_normalized, cmap='gray')
    plt.title(f"Slice {i+1}")
    plt.axis('off')  # Turn off axis
    plt.show()

    # Construct the full file path for destination
    destination_file_path = os.path.join(output_path, f"slice_{i+1}.png")

    # Save the image to the destination directory
    plt.imsave(destination_file_path, slice_data_normalized, cmap='gray')