import os
import sys
import shutil
import csv

def copy_files(file_list_path, output_dir, input_dirs, og):
    with open(file_list_path, 'r') as file:
        file_names = [os.path.basename(line.strip()) for line in file if line.strip()]
    
    os.makedirs(output_dir, exist_ok=True)
    letters = 65 #A
    combination = []
    for input_dir in input_dirs:
        combination.append([input_dir, chr(letters)])
        for file in file_names:
            if og:
                to_be_found = os.path.basename(file)
            else:
                to_be_found = os.path.basename(file).replace(".jpg", "_out.jpg")
            for root, _, files in os.walk(input_dir):
                for file_name in files:
                    if file_name == to_be_found:
                        source_file = os.path.join(root, file_name)
                        destination_path = os.path.join(output_dir, os.path.basename(file_name).replace(".jpg", "_" + chr(letters) + ".jpg"))
                        shutil.copy(source_file, destination_path)
        letters += 1
    
    with open(os.path.join(output_dir, "combination.txt"), "w", newline="") as file:
        writer = csv.writer(file, delimiter=" ")
        writer.writerows(combination)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py <og/gen> <file_list.txt> <output_dir> <input_dir1> [<input_dir2> ...]")
        sys.exit(1)

    og_flag = (sys.argv[1] == "og")
    file_list = sys.argv[2]
    output_dir = sys.argv[3]
    input_dirs = sys.argv[4:]

    copy_files(file_list, output_dir, input_dirs, og_flag)
