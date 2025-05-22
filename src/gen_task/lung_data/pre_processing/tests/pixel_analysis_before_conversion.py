import pydicom
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

def normalizePlanes(pixel_data, maxHU=400., minHU=-1000.):
    # Normalize function to map HU values to range [0, 1]
    normalized_pixel_data = (pixel_data - minHU) / (maxHU - minHU)
    normalized_pixel_data[normalized_pixel_data < 0] = 0
    normalized_pixel_data[normalized_pixel_data > 1] = 1
    return normalized_pixel_data

def apply_contrast_enhancement(image):
    # Histogram equalization
    equalized_image = exposure.equalize_hist(image)

    # Gamma correction (adjust gamma value as needed)
    gamma_corrected_image = exposure.adjust_gamma(equalized_image, gamma=1.2)

    return gamma_corrected_image

def apply_windowing(image, window_center, window_width):
    # Perform windowing
    windowed_image = np.clip(image, window_center - window_width / 2, window_center + window_width / 2)

    # Normalize the windowed image
    windowed_image = (windowed_image - (window_center - 0.5)) / window_width

    return windowed_image

def analyze_dicom_file(dicom_filepath):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_filepath)

    # Apply normalization to pixel data
    normalized_pixel_data = normalizePlanes(dicom_data.pixel_array)

    # Apply contrast enhancement
    enhanced_image = apply_contrast_enhancement(normalized_pixel_data)

    # Apply lung windowing (adjust window center and width as needed)
    lung_window_center = -600
    lung_window_width = 1500
    lung_windowed_image = apply_windowing(enhanced_image, lung_window_center, lung_window_width)

    # Plot both original and processed images
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_pixel_data, cmap='gray')
    plt.title('Original DICOM Image')
    plt.axis('off')

    # Processed image
    plt.subplot(1, 2, 2)
    plt.imshow(lung_windowed_image, cmap='gray')
    plt.title('Processed DICOM Image')
    plt.axis('off')

    plt.show()

# Example usage
dicom_filepath = "1-030.dcm"
analyze_dicom_file(dicom_filepath)