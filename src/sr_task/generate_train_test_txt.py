import os
import sys
import random

def list_image_files(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                image_files.append(path)
    return image_files

def save_paths_to_txt(file_paths, output_file):
    with open(output_file, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')

def split_train_test(file_paths, train_percentage):
    random.shuffle(file_paths)
    split_index = int(len(file_paths) * train_percentage)
    train_paths = file_paths[:split_index]
    test_paths = file_paths[split_index:]
    return train_paths, test_paths

if len(sys.argv) != 3:
    print("Usage: python generate_test_train_txt.py input_directory train_percentage")
    sys.exit(1)

input_directory = sys.argv[1]
train_percentage = float(sys.argv[2])

image_files = list_image_files(input_directory)

full_file = os.path.join(input_directory, 'full+.txt')
train_file = os.path.join(input_directory, 'train+.txt')
test_file = os.path.join(input_directory, 'test+.txt')

save_paths_to_txt(image_files, full_file)

train_paths, test_paths = split_train_test(image_files, train_percentage)

save_paths_to_txt(train_paths, train_file)
save_paths_to_txt(test_paths, test_file)

print("Image paths saved:")
print("Full:", full_file)
print("Train:", train_file)
print("Test:", test_file)
