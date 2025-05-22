import os
import sys
from PIL import Image

def downscale(input_dir, output_dir, factor):
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    subdir_count = 0

    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file_name)
                output_subdir = os.path.join(output_dir, f"subdir_{subdir_count}")

                if count % 2500 == 0:
                    subdir_count += 1
                    output_subdir = os.path.join(output_dir, f"subdir_{subdir_count}")
                    os.makedirs(output_subdir, exist_ok=True)

                output_path = os.path.join(output_subdir, os.path.relpath(input_path, input_dir))

                img = Image.open(input_path)

                new_width = int((img.width * factor) // 1)
                new_height = int((img.height * factor) // 1)

                downscaled_img = img.resize((new_width, new_height), Image.LANCZOS)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                downscaled_img.save(output_path)

                count += 1

if len(sys.argv) != 4:
    print("Usage: python downscale.py input_directory output_directory downscale_factor")
    sys.exit(1)

input_directory = sys.argv[1]
output_directory = sys.argv[2]
factor = float(sys.argv[3])

downscale(input_directory, output_directory, factor)
