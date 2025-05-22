import os
import shutil
import sys

def process_files(directory, suffix, file_path):
    filename = os.path.basename(file_path) #32.jpg
    filename = filename.replace(".jpg", "_out.jpg") #32_out.jpg
    full_path = os.path.join(directory, filename) # ././visualization/32_out.jpg
    shutil.copy(full_path, new_directory)
    new_name = filename.replace("out.jpg", suffix + ".jpg")
    os.rename(os.path.join(new_directory, filename), os.path.join(new_directory, new_name))

if (len(sys.argv) % 2) != 0:
    print("Usage: python analysis.txt <dir suffix>")
    sys.exit(1)

directories = []
suffixes = []
file_paths_file = sys.argv[1]

for i in range(1, len(sys.argv)):
    if i % 2 == 0:
        directories.append(sys.argv[i])
    else:
        suffixes.append(sys.argv[i])


file_paths = []

with open(file_paths_file, "r") as file:
    for line in file:
        file_paths.append(line.strip())

new_directory = "file_analysis"
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

for file_path in file_paths:
    for directory, suffix in zip(directories, suffixes):
        process_files(directory, suffix, file_path)
