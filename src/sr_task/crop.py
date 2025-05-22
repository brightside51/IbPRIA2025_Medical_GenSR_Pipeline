import cv2
import numpy as np
import sys
import os

def crop_image(image, top, bottom, left, right):
    cropped_image = image[top:bottom+1, left:right+1]
    return cropped_image

def find_item(list, item):
    for sublist in list:
        if sublist[0] == item:
            return sublist[1], sublist[2], sublist[3], sublist[4]

if len(sys.argv) != 4:
    print("Usage: python crop_txt.py <solution.txt> input_directory output_directory")
    sys.exit(1)

solution = sys.argv[1]
input_directory = sys.argv[2]
output_directory = sys.argv[3]

files = os.listdir(input_directory)

paths = []
with open(solution, "r") as file1:
    for line in file1:
        items = line.strip().split(",")
        paths.append([items[0], int(items[1]), int(items[2]), int(items[3]), int(items[4])])


for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.jpg'):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_directory)
            output_path = os.path.join(output_directory, relative_path, file)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            solution_dir = os.path.relpath(input_path, input_directory)
            image = cv2.imread(input_path)
            top, bottom, left, right = find_item(paths, solution_dir)
            
            cropped_image = crop_image(image, top, bottom, left, right)
            cv2.imwrite(output_path, cropped_image)


print("Cropped image saved successfully.")
