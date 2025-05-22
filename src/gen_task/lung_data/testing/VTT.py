import os
import random
import shutil

def get_all_images_from_directory(root_dir):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add any other image extensions if needed
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def copy_and_rename_images(image_paths, output_dir, prefix):
    for idx, img_path in enumerate(image_paths):
        img_ext = os.path.splitext(img_path)[1]
        new_name = f"{prefix}_{idx + 1}{img_ext}"
        shutil.copy(img_path, os.path.join(output_dir, new_name))
        yield new_name, img_path


# Directory to save the sampled images
output_dir = '/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/HA-GAN/scripts/VTT'  # Replace with the path to your output directory
os.makedirs(output_dir, exist_ok=True)

# Directories containing the real and generated images
generated_images_dir = '/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/HA-GAN/scripts/128/first_try_80k/Generated_50_128'  # Replace with the path to your real images directory
real_images_dir = '/nas-ctm01/homes/jamartins/joao-a-martins/joao-a-martins-msc/LIDC/joao_processing/HA-GAN/scripts/128/first_try_80k/Real_50_128'  # Replace with the path to your generated images directory


# Get all images from these directories
real_images = get_all_images_from_directory(real_images_dir)
generated_images = get_all_images_from_directory(generated_images_dir)

# Ensure both lists have enough images
total_images = 50
min_ratio = 0.4
max_ratio = 0.6

if len(real_images) < total_images * min_ratio or len(generated_images) < total_images * min_ratio:
    raise ValueError("Each directory must contain at least 40% of the total images")

# Randomly choose the number of real images (between 40% and 60% of total_images)
num_real_images = random.randint(int(total_images * min_ratio), int(total_images * max_ratio))
num_generated_images = total_images - num_real_images

# Randomly sample the chosen number of images from each list
sampled_real_images = random.sample(real_images, num_real_images)
sampled_generated_images = random.sample(generated_images, num_generated_images)

# Combine the sampled images
final_sample = sampled_real_images + sampled_generated_images

# Shuffle the final sample to mix real and generated images
random.shuffle(final_sample)

# Copy sampled images to the output directory and log their types and original paths
log_file_path = os.path.join(output_dir, 'image_log.txt')
with open(log_file_path, 'w') as log_file:
    for new_name, original_path in copy_and_rename_images(final_sample, output_dir, "img"):
        if original_path in sampled_real_images:
            base_dir_name = os.path.basename(real_images_dir)
            image_type = 'real'
        else:
            base_dir_name = os.path.basename(generated_images_dir)
            image_type = 'generated'
        
        relative_path = os.path.relpath(original_path, start=os.path.dirname(os.path.dirname(original_path)))
        log_file.write(f"New Name: {new_name}, Original Path: {os.path.join(base_dir_name, relative_path)}, Type: {image_type}\n")

print(f"Sampled images have been saved to {output_dir}")
print(f"A log of the image types has been saved to {log_file_path}")
