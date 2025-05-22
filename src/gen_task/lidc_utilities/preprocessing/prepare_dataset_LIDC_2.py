import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from aux_open_images import get_pixels_hu, load_scan, normalize,crop_scan, resample3d

warnings.filterwarnings(action='ignore')

"""
Code to pre-process the dataset: 
Crop 2D Nodule
"""
"""
#Get Directory setting
DICOM_DIR = "C:/Users/guigo/Documents/Databases/LIDC/LIDC-IDRI"
LIDC_IDRI_NODULE ="C:/Users/guigo/Documents/Databases/LIDC/"
MASK_DIR = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Mask/"
CLEAN_DIR_MASK = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Clean_Mask/"
META_DIR = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Meta/"
NODULE_DIR = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Nodule_2D/"
"""
DICOM_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI"
LIDC_IDRI_NODULE ="/nas-ctm01/datasets/private/LUCAS/lidc/"
MASK_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Nodule_Mask/"
CLEAN_DIR_MASK = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Clean_Mask/"
META_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Meta/"
NODULE_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Nodule_2D/"

size_box=50
classes="binary"
center="estimated"

if not os.path.exists(NODULE_DIR):
    os.makedirs(NODULE_DIR)

cases=pd.read_csv(META_DIR+"/meta_central.csv")

if(classes=="binary"):
    cases=cases.loc[cases["is_cancer"] != "Ambiguous"]            

images_path=cases["dicom_path"].to_list()

case_ids=cases["patient_id"].to_list()
nodule_ids=cases["nodule_no"].to_list()
slice_ids=cases["slice_no_inv"].to_list()

if(classes=="binary"):
    l=cases["is_cancer"].to_list()
    labels= l
       
else:
    l=cases["malignacy"].to_list()
    labels= [int(i) for i in l]

if(center=="ground_truth"):
    x=cases["x_nod"].to_list()
    y=cases["y_nod"].to_list()
else:
    x=cases["x_nod_est"].to_list()
    y=cases["y_nod_est"].to_list()

slice_thickness=cases["slice_thickness"].to_list()
pixel_spacing=cases["pixel_spacing"].to_list()

for idx in range(len(cases)):
    print("Processing "+str(case_ids[idx])+" "+str(nodule_ids[idx]))
    #path=images_path[idx].replace("/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI",
    #                           "C:/Users/guigo/Documents/Databases/LIDC/LIDC-IDRI")
    print(images_path[idx])
    #Lista com os slices lidos
    ct_scan=load_scan(images_path[idx])
    ct_scan_image = get_pixels_hu(ct_scan)
    
    print(ct_scan[0].SliceThickness, ct_scan[0].PixelSpacing[0], ct_scan[0].PixelSpacing[1])
    print(slice_thickness[idx], pixel_spacing[idx], pixel_spacing[idx])
    """
    plt.figure()
    plt.imshow(ct_scan_image[slice_ids[idx],:,:])
    plt.title(str(case_ids[idx])+"__"+str(nodule_ids[idx]))
    plt.scatter(x[idx],y[idx])
    """
    
    s=slice_thickness[idx]
    p=pixel_spacing[idx]
    ct_scan_image, (xx,yy,zz)=resample3d(ct_scan_image, ct_scan,[s,p,p], [1,1,1])
    
    X=int(x[idx]*p)
    Y=int(y[idx]*p)
    Z=int(slice_ids[idx]*s)
    """
    plt.figure()
    plt.imshow(ct_scan_image[Z,:,:])
    plt.title(str(case_ids[idx])+"__"+str(nodule_ids[idx]))
    plt.scatter(X,Y)
    """
    
    ct_scan_image = np.array(ct_scan_image, dtype=np.int16)
    ct_scan_image=normalize(ct_scan_image)
    ct_scan_image = ct_scan_image.astype(np.float32)
    
    nodule=crop_scan(ct_scan_image,Z,X,Y, size_box=size_box)
    """
    plt.figure()
    plt.imshow(nodule)
    """
      
    outfile=NODULE_DIR+str(case_ids[idx])+"__"+str(nodule_ids[idx])+".npy"
    np.save(outfile, nodule)
    """
    n=np.load(outfile)
    plt.figure()
    plt.imshow(n)
    plt.title(str(case_ids[idx])+"__"+str(nodule_ids[idx]))
    """
