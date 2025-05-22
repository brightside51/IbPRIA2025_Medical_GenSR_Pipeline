
import os
import numpy as np
import pydicom
import json

def convert_dicom_to_npy(input_dir, output_dir):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a dictionary to store DICOM metadata
    metadata_dict = {}
    print("COMECEI")
    # Traverse through patient directories
    for patient_dir in os.listdir(input_dir):
        patient_dir_path = os.path.join(input_dir, patient_dir)
        print(patient_dir_path)
        if not os.path.isdir(patient_dir_path):
            continue  # Skip if not a directory
        for date_dir in os.listdir(patient_dir_path):
            date_dir_path = os.path.join(patient_dir_path, date_dir)
            print(date_dir_path)

            if not os.path.isdir(date_dir_path):
                continue  # Skip if not a directory
            for final_dir in os.listdir(date_dir_path):
                final_dir_path = os.path.join(date_dir_path, final_dir)
                print(final_dir_path)

                if not os.path.isdir(final_dir_path):
                    continue  # Skip if not a directory   

                # List DICOM files for the current patient
                dicom_files = [f for f in os.listdir(final_dir_path) if f.endswith('.dcm')]
                if(len(dicom_files) < 10):
                    continue
                for file_name in dicom_files:
                    temp_file_name = file_name
                    file_path = os.path.join(final_dir_path, file_name)
                    
                    # Read DICOM file
                    dicom_data = pydicom.dcmread(file_path)
                    
                    # Convert pixel data to numpy array
                    pixel_array = dicom_data.pixel_array

                    # Save pixel array as .npy file
                    npy_file_path = os.path.join(output_dir, patient_dir, file_name.replace('.dcm', '.npy'))
                    os.makedirs(os.path.dirname(npy_file_path), exist_ok=True)
                    np.save(npy_file_path, pixel_array)

                    # Convert DICOM metadata to dictionary
                    dicom_dict = {}
                    for elem in dicom_data:
                        if elem.VR != 'SQ':
                            dicom_dict[str(elem.tag)] = str(elem.value)
                        else:
                            # Handle sequences separately
                            dicom_dict[str(elem.tag)] = []
                            for seq_item in elem.value:
                                seq_dict = {}
                                for seq_elem in seq_item:
                                    seq_dict[str(seq_elem.tag)] = str(seq_elem.value)
                                dicom_dict[str(elem.tag)].append(seq_dict)

                    json_data = json.dumps(dicom_dict, indent=4)
                    # Save JSON data to a file
                    json_file_path = os.path.join(output_dir, patient_dir, temp_file_name.replace('.dcm', '.json'))
                    with open(json_file_path, "w") as json_file:
                        json_file.write(json_data)
                    # Save metadata as JSON file

    print("Conversion complete.")

# Example usage:
input_directory = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"
output_directory = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/processed/images"
convert_dicom_to_npy(input_directory, output_directory)
