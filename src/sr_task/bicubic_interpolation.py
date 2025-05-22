import os
import sys
import cv2

def bicubic_interpolation(input_txt, input_base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_txt, 'r') as file:
        image_paths = file.read().splitlines()

    for path in image_paths:

        input_path = os.path.join(input_base_dir, path)

        output_path = os.path.join(output_dir, path).replace(".jpg", "_out.jpg")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        image = cv2.imread(input_path)

        if image is not None:
            height, width = image.shape[:2]
            new_height, new_width = height * 4, width * 4
            interpolated_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(output_path, interpolated_image)
            print(f"Interpolated image saved as '{output_path}'")
        else:
            print(f"Failed to load image from '{input_path}'")

if len(sys.argv) != 4:
    print("Usage: python bicubic_interpolation.py input_paths.txt input_base_directory output_directory")
    sys.exit(1)

bicubic_interpolation(sys.argv[1], sys.argv[2], sys.argv[3])
