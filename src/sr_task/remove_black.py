from PIL import Image
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python remove_black.py <input>")
    sys.exit(1)

def is_all_black(image):
    grayscale_image = image.convert("L")
    return all(pixel == 0 for pixel in grayscale_image.getdata())

def remove_black_images(directory):
    counter = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and (item.endswith('.jpg') or item.endswith('.png')):
            with Image.open(item_path) as img:
                if is_all_black(img):
                    os.remove(item_path)
                    counter += 1
    return counter

data_directory = sys.argv[1]
counter = remove_black_images(data_directory)
print("Removed " + str(counter) + " images")