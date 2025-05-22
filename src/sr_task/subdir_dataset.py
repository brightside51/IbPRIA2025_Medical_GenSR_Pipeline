import os
import shutil
import sys

if len(sys.argv) != 2:
    print("Usage: python subdir_dataset.py <directory>")
    sys.exit(1)

source_directory = sys.argv[1]

def distribute_files(source_dir):
    file_count = 0
    subdir_count = 0
    target_dir = source_dir

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        if file_count % 2500 == 0:
            subdir_count += 1
            new_subdir = os.path.join(target_dir, f"subdir_{subdir_count}")
            os.makedirs(new_subdir, exist_ok=True)
        dest_path = os.path.join(new_subdir, filename)
        shutil.move(source_path, dest_path)
        file_count += 1

    print(f"Total files moved: {file_count}")


distribute_files(source_directory)
