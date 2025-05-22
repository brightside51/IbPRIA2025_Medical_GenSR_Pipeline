import os
import random
from PIL import Image

def select_middle_slice(input_dir, count, type_label):
    selected_images = []
    
    # Check if it's a directory
    if os.path.isdir(input_dir):
        # Iterate through each subdirectory (sample) in the input directory
        for sample_name in os.listdir(input_dir):
            sample_path = os.path.join(input_dir, sample_name)
            if os.path.isdir(sample_path):
                # List and sort all files in the sample folder
                files = [f for f in os.listdir(sample_path) if os.path.isfile(os.path.join(sample_path, f))]
                files.sort()
                
                # Select the middle section of the sorted list
                if files:
                    middle_start = len(files) // 5
                    middle_end = len(files) - middle_start
                    middle_files = files[middle_start:middle_end]
                    
                    # Randomly select a file from the middle section
                    if middle_files:
                        selected_file = random.choice(middle_files)
                        selected_file_path = os.path.join(sample_path, selected_file)
                        selected_images.append((selected_file_path, count, type_label))
                        count += 1

    return selected_images, count

def create_set(input_dirs, output_dir, real_ratio):
    # Calculate the number of real and fake images needed
    total_images = 30
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    selected_images = []
    real_selected_images, count = select_middle_slice(input_dirs['real'], count, 'real')
    fake_selected_images_1, count = select_middle_slice(input_dirs['fake'][0], count, 'generated')
    fake_selected_images_2, count = select_middle_slice(input_dirs['fake'][1], count, 'generated')
    fake_selected_images_3, count = select_middle_slice(input_dirs['fake'][2], count, 'generated')

    fake_selected_images_list = [fake_selected_images_1, fake_selected_images_2, fake_selected_images_3]
    
    r = 0
    f = [0, 0, 0]

    for i in range(total_images):
        if random.random() <= real_ratio and r < len(real_selected_images):
            # Select a real image
            image_path, _, type_label = real_selected_images[r]
            r += 1
        else:
            # Select a fake image from a random fake image list
            fake_index = random.choice([0, 1, 2])
            fake_images = fake_selected_images_list[fake_index]
            if f[fake_index] < len(fake_images):
                image_path, _, type_label = fake_images[f[fake_index]]
                f[fake_index] += 1
            else:
                # Fallback in case one of the fake lists runs out of images
                for j in range(3):
                    if f[j] < len(fake_selected_images_list[j]):
                        image_path, _, type_label = fake_selected_images_list[j][f[j]]
                        f[j] += 1
                        break

        image = Image.open(image_path)
        output_file_path = os.path.join(output_dir, f"slice_{i + 1}_{type_label}.png")
        image.save(output_file_path)

if __name__ == "__main__":
    input_directories = {
        'real': r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\SHOW\HA_GAN_cross\gen_64_duplicate_cross\Real_50",
        'fake': [
            r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\SHOW\HA_GAN_cross\gen_64_duplicate_cross\Generated_50",
            r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\SHOW\HA_GAN_cross\gen_64_duplicate_cross\Generated_50_2",
            r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\SHOW\HA_GAN_cross\gen_64_duplicate_cross\Generated_50_HAGAN"
        ]
    }

    output_directories = [
        (r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\VTT\Set_1", 0.7),  # 70% real, 50% fake
        (r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\VTT\Set_2", 0.5),  # 30% real, 70% fake
        (r"C:\Users\Diogo Campas\Desktop\Dissertação_23_24\code\VTT\Set_3", 0.3)   # 70% real, 30% fake
    ]

    for output_dir, real_ratio in output_directories:
        create_set(input_directories, output_dir, real_ratio)
        #print(f"Order for {output_dir}: {order_list}")
