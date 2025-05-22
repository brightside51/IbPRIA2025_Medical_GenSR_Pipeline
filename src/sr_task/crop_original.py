import cv2
import numpy as np
import sys
import os

def find_non_black_pixels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows, cols = gray.shape
    top, bottom, left, right = None, None, None, None

    for y in range(rows):
        for x in range(cols):
            if gray[y,x] > 10:
                top = y
                break
        if top is not None:
            break


    for y in range(rows-1, -1, -1):
        for x in range(cols):
            if gray[y,x] > 10:
                bottom = y
                break
        if bottom is not None:
            break

    for x in range(cols):
        for y in range(rows):
            if gray[y,x] > 10:
                left = x
                break
        if left is not None:
            break

    for x in range(cols-1, -1, -1):
        for y in range(rows):
            if gray[y,x] > 10:
                right = x
                break
        if right is not None:
            break
    return top, bottom, left, right

def crop_image(image, top, bottom, left, right):
    cropped_image = image[top:bottom+1, left:right+1]
    return cropped_image

if len(sys.argv) != 3:
    print("Usage: python crop.py input_directory output_directory")
    sys.exit(1)

input_directory = sys.argv[1]
output_directory = sys.argv[2]
files = os.listdir(input_directory)
solution1 = []
solution2 = []

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.jpg'):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_directory)
            output_path = os.path.join(output_directory, relative_path, file)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            solution = os.path.relpath(input_path, input_directory)
            image = cv2.imread(input_path)
            top, bottom, left, right = find_non_black_pixels(image)
            cropped_image = crop_image(image, top, bottom, left, right)
            solution1.append([solution.replace(".jpg", "_out.jpg"), top, bottom, left, right])
            solution2.append([os.path.basename(input_path).replace(".jpg", "_out.jpg"), top, bottom, left, right])
            cv2.imwrite(output_path, cropped_image)

with open("same_structure_crop.txt", "w") as file1:
    for sublist in solution1:
        line = ",".join(map(str, sublist))
        file1.write(line + "\n")

with open("crop.txt", "w") as file2:
    for sublist in solution1:
        line = ",".join(map(str, sublist))
        file2.write(line + "\n")

print("Cropped image saved successfully.")
