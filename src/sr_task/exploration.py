import os
from PIL import Image
import numpy as np
import sys
import remove_black

def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        return img.size

def get_all_image_resolutions(directory):
    resolutions_count = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            resolution = get_image_resolution(image_path)
            resolution_str = f"{resolution[0]}x{resolution[1]}"
            if resolution_str in resolutions_count:
                resolutions_count[resolution_str] += 1
            else:
                resolutions_count[resolution_str] = 1
    return resolutions_count

def calculate_combined_stats_of_pixel_values(directory):
    all_pixel_values = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                pixel_values = np.array(img)
                all_pixel_values.extend(pixel_values.flatten())
    
    print("Combined Statistics of Pixel Color Values:")
    print("Mean:", np.mean(all_pixel_values))
    print("Median:", np.median(all_pixel_values))
    print("Standard Deviation:", np.std(all_pixel_values))

if len(sys.argv) != 2:
    print("Usage: python exploration.py <input>")
    sys.exit(1)



data_directory = sys.argv[1]


counter = remove_black.remove_black_images(data_directory)
print("Removed " + str(counter) + " images")

image_resolutions = get_all_image_resolutions(data_directory)

for resolution, count in image_resolutions.items():
    print(f"{resolution}: {count}")

calculate_combined_stats_of_pixel_values(data_directory)