# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:22:13 2023

@author: guigo
"""
import pydicom
import os
import numpy as np 
import scipy.ndimage
import matplotlib.pyplot as plt

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s, force=True) for s in os.listdir(path) if s.endswith('.dcm')]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def crop_scan(ct_scan_complete, nodule_slice, x, y, size_box):
    ct_scan=ct_scan_complete[nodule_slice,:,:]
    top_left = (max(x - size_box//2,0), max(y - size_box//2,0))
    max_x=len(ct_scan[:,0])-1
    max_y=len(ct_scan[0,:])-1    
    bottom_right = (min(x + size_box//2,max_x),min(y + size_box//2,max_y))
    cropped_scan = ct_scan[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if len(cropped_scan[0,:])<size_box:
            column_to_be_added=np.zeros((len(cropped_scan[:,0]),size_box-len(cropped_scan[0,:])), dtype=float)
            cropped_scan = np.append(cropped_scan[:,:], column_to_be_added, axis=1)
    if len(cropped_scan[:,0])<size_box:
            column_to_be_added=np.zeros((size_box-len(cropped_scan[:,0]),size_box), dtype=float)
            cropped_scan = np.append(cropped_scan[:,:], column_to_be_added, axis=0)
    return cropped_scan

def normalize(scan, minHU = -1000, maxHU = 400):
  scan = (scan - minHU) / (maxHU - minHU)
  scan[scan >= 1] = 1
  scan[scan <= 0] = 0
  return scan

# Code adapted from:
# https://github.com/Kri-Ol/DICOM-Resampler/blob/master/Resampling3D.ipynb

def resample3d(image, scan,old_spacing, new_spacing):
    # Determine current pixel spacing
    spacing = old_spacing     
    resize_x    = spacing[1] / new_spacing[1]
    new_shape_x = np.round(image.shape[1] * resize_x)
    resize_x    = float(new_shape_x) / float(image.shape[1])
    sx = spacing[1] / resize_x

    resize_y    = spacing[2] / new_spacing[2]
    new_shape_y = np.round(image.shape[2] * resize_y)
    resize_y    = new_shape_y / image.shape[2]
    sy = spacing[2] / resize_y

    resize_z    = spacing[0] / new_spacing[0]
    new_shape_z = np.round(image.shape[0] * resize_z)
    resize_z    = float(new_shape_z) / float(image.shape[0])
    sz = spacing[0] / resize_z
    
    image = scipy.ndimage.interpolation.zoom(image, (resize_z,resize_x, resize_y), order=1)
    
    return (image, (sz,sx, sy))

def resample2d(image, scan, new_spacing):
    # Determine current pixel spacing
    spacing = map(float, [scan[0].PixelSpacing[0], scan[0].PixelSpacing[1], scan[0].SliceThickness])
    spacing = np.array(list(spacing))

    resize_x    = spacing[0] / new_spacing[0]
    new_shape_x = np.round(image.shape[0] * resize_x)
    resize_x    = float(new_shape_x) / float(image.shape[0])
    sx = spacing[0] / resize_x

    resize_y    = spacing[1] / new_spacing[1]
    new_shape_y = np.round(image.shape[1] * resize_y)
    resize_y    = new_shape_y / image.shape[1]
    sy = spacing[1] / resize_y
    
    image = scipy.ndimage.interpolation.zoom(image, (resize_x, resize_y, 1.0), order=1)
    
    return (image, (sx, sy))
