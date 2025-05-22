import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import napari
import re
# Define the paths to the directories containing the images
dir1 = 'VTT_final_final'
dir2 = 'images'

# Function to load and sort images from a directory
def load_images_from_directory(directory):
    images = []
    filenames = sorted(os.listdir(directory), key=lambda x: int(re.findall(r'\d+', x)[0]))
    for filename in filenames:
        if filename.endswith('.png'):  # Adjust this if your images have a different extension
            img = io.imread(os.path.join(directory, filename))
            images.append(img)
    return np.stack(images, axis=0)

# Load images from both directories
volume1 = load_images_from_directory(dir1)
volume2 = load_images_from_directory(dir2)

# Volumetric rendering using napari
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(volume1, name='Volume 1', colormap='gray')
    # viewer.add_image(volume2, name='Volume 2', colormap='magma')