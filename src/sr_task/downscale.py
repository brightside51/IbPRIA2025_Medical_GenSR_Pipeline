import os
import sys
from PIL import Image, ImageFilter

def downscale(input_dir, output_dir, factor):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file_name)
                output_subdir = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, output_subdir, file_name)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                img = Image.open(input_path)

                blur_img = img.filter(ImageFilter.BLUR)
                blur_img1 = blur_img.filter(ImageFilter.BLUR)

                new_width = int(round(img.width * factor))
                new_height = int(round(img.height * factor))

                downscaled_img = blur_img1.resize((new_width, new_height), Image.LANCZOS)
                downscaled_img.save(output_path)

if len(sys.argv) != 4:
    print("Usage: python downscale.py input_directory output_directory downscale_factor")
    sys.exit(1)

input_directory = sys.argv[1]
output_directory = sys.argv[2]
factor = float(sys.argv[3])

downscale(input_directory, output_directory, factor)
