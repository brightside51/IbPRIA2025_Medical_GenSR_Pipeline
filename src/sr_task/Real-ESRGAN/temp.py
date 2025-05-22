import os

def find_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.relpath(os.path.join(root, file), directory) + ", " + os.path.relpath(os.path.join(root, file), directory))
    
    return image_paths

def write_paths_to_file(image_paths, output_file):
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')

if __name__ == "__main__":
    directory = '/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1/pspereira'  # Change this to the directory containing your images
    output_file = 'image_paths.txt'  # Change this to the desired output file name
    
    image_paths = find_images(directory)
    write_paths_to_file(image_paths, output_file)
    print("Image paths have been written to", output_file)
