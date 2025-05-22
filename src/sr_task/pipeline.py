import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import cv2
import numpy as np
from pathlib import Path
import os

def _pixel_process(ds, pixel_array):
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        rescale_slope = float(ds.RescaleSlope)
        rescale_intercept = float(ds.RescaleIntercept)
        pixel_array = pixel_array * rescale_slope + rescale_intercept
    else:
        pixel_array = apply_modality_lut(pixel_array, ds)

    if 'VOILUTFunction' in ds and ds.VOILUTFunction == 'SIGMOID':
        pixel_array = apply_voi_lut(pixel_array, ds)
    elif 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth
        if type(window_center) == pydicom.multival.MultiValue:
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
        if type(window_width) == pydicom.multival.MultiValue:
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
        pixel_array = _get_LUT_value_LINEAR_EXACT(pixel_array, window_width, window_center)
    else:
        pixel_array = apply_voi_lut(pixel_array, ds)

    pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())) * 255.0

    if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    return pixel_array.astype('uint8')

def _get_LUT_value_LINEAR_EXACT(data, window, level):
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    data = np.piecewise(data, 
        [data <= (level - (window) / 2),
         data > (level + (window) / 2)],
        [data_min, data_max, lambda data: ((data - level + window / 2) / window * data_range) + data_min])
    return data

def _is_unsupported(ds):
    try:
        if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.104.1':
            return 'Encapsulated PDF Storage'
        elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.59':
            return 'Key Object Selection Document'
    except:
        pass
    return False

def dicom_to_jpg(dicom_file, output_dir, base_input_dir):
    dicom_file = Path(dicom_file)
    output_dir = Path(output_dir)

    ds = pydicom.dcmread(dicom_file, force=True)
    
    if _is_unsupported(ds):
        print(f"'{dicom_file}' is not supported")
        return

    pixel_array = ds.pixel_array.astype(float)
    
    pixel_array = _pixel_process(ds, pixel_array)
    
    relative_path = dicom_file.relative_to(base_input_dir)
    output_file = output_dir / relative_path.with_suffix('.jpg')
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_file), pixel_array)

def process_directory(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for dicom_file in input_dir.rglob('*.dcm'):
        dicom_to_jpg(dicom_file, output_dir, input_dir)
        print(f"Converted {dicom_file} to JPEG and saved to {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)
    print(f"Processed all DICOM files in {input_dir} and saved JPEGs to {output_dir}")
