import os
import statistics
import json


def count_dicom_files(directory):
    """
    Count the number of DICOM files in a directory.
    """
    dicom_count = 0
    dicom_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                # Split the filename into base and extension
                base, ext = os.path.splitext(file)
                if '1' not in base.split('-')[0]:
                    continue 
                dicom_count += 1
                dicom_files.append(os.path.join(root, file))
    return dicom_count, dicom_files

def find_max_and_min_dicom_files(main_directory):
    """
    Find the directory with the maximum and minimum number of DICOM files and return their names and counts.
    Also return the average number of DICOM files, median number of DICOM files, 
    and counts above and below the average.
    """
    max_count = 0
    max_directory = None
    min_count = float('inf')
    min_directory = None
    dicom_counts = []

    for root, dirs, files in os.walk(main_directory):
        for directory in dirs:
            if directory.startswith('LIDC'):
                dicom_count, _ = count_dicom_files(os.path.join(root, directory))
                dicom_counts.append(dicom_count)
                if dicom_count > max_count and dicom_count >= 10:
                    max_count = dicom_count
                    max_directory = os.path.join(root, directory)
                if dicom_count < min_count and dicom_count >= 10:
                    min_count = dicom_count
                    min_directory = os.path.join(root, directory)

    average_dicom_files = sum(dicom_counts) / len(dicom_counts)
    median_dicom_files = statistics.median(dicom_counts)
    above_average = sum(1 for count in dicom_counts if count > average_dicom_files)
    below_average = sum(1 for count in dicom_counts if count < average_dicom_files)
    
    return max_directory, max_count, min_directory, min_count, average_dicom_files, median_dicom_files, above_average, below_average

# Specify the main directory containing child directories with DICOM files
main_directory = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"

# Find the directory with the maximum and minimum number of DICOM files and median DICOM files
max_directory, max_dicom_files, min_directory, min_dicom_files, average_dicom_files, median_dicom_files, above_average, below_average = find_max_and_min_dicom_files(main_directory)

print("Directory with the maximum number of DICOM files:", max_directory)
print("Maximum number of DICOM files:", max_dicom_files)

print("Directory with the minimum number of DICOM files:", min_directory)
print("Minimum number of DICOM files:", min_dicom_files)

print("Average number of DICOM files:", average_dicom_files)
print("Median number of DICOM files:", median_dicom_files)
print("Number of DICOM files above the average:", above_average)
print("Number of DICOM files below the average:", below_average)

# Create a dictionary to store the information
result = {
    "max_directory": max_directory,
    "max_dicom_files": max_dicom_files,
    "min_directory": min_directory,
    "min_dicom_files": min_dicom_files,
    "average_dicom_files": average_dicom_files,
    "median_dicom_files": median_dicom_files,
    "above_average": above_average,
    "below_average": below_average
}

# Specify the file path for saving the JSON data
output_file_path = "dicom_info.json"

# Write the information to a JSON file
with open(output_file_path, "w") as json_file:
    json.dump(result, json_file, indent=4)

print("Information saved to", output_file_path)
