import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
from pylidc.utils import consensus
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.measure import find_contours

warnings.filterwarnings(action='ignore')

"""
Code to pre-process the dataset: 
Compute average malignacy and save the masks for the slices with tummors
Split and train, validation and test

Based on code from: https://github.com/jaeho3690/LIDC-IDRI-Preprocessing/tree/master

The masks for the lungs are not computed

Uses:  MedPy==0.4.0 ; pylidc==0.2.1 ; numpy==1.23.5
"""

"""
This python script will create the image, mask files and save them to the data folder. 
The script will also create a meta_info.csv file containing information about whether the nodule is cancerous. 
In the LIDC Dataset, each nodule is annotated at a maximum of 4 doctors. 
Each doctors have annotated the malignancy of each nodule in the scale of 1 to 5. 
I have chosed the median high label for each nodule as the final malignancy. 
The meta_csv data contains all the information and will be used later in the classification stage. 
Running this script will output .npy files for each slice with a size of 512*512


Columns meta:
patient_id - patient identification (number)
nodule_no - nodule identification (for paatients with more than one nodule)
slice_no - to match the scan when read with pylidc
slice_no_inv - to match scan when read normaly
slice_mask_no - to match masks created with pylidc
original_image -  name original image
mask_image - name original mask
malignancy - "average" (from 0 to 5)
is_cancer - binary (0 or 1)
is_clean - true for slices without nodule
'x_nod','y_nod' - nodule center read from the list3.2.csv file 
https://www.via.cornell.edu/lidc/
[nodule identified by the slice number in the interval slice-2 to slice+2]
'x_nod_est','y_nod_est' - nodule center estimated from mask in each slice
data_split - indication of the set (train, validation, test)

Columns meta_central (just the central slices for 2d):
same info

Makes a train/ val/ test split. 
This will create an additional clean_meta.csv, meta.csv containing information about the nodules, 
train/val/test split.

A nodule may contain several slices of images. 
Some researches have taken each of these slices indpendent from one another. 
However, I believe that these image slices should not be seen as independent from adjacent slice image. 
Thus, I have tried to maintain a same set of nodule images to be included in the same split. 
Although this apporach reduces the accuracy of test results, it seems to be the honest approach.
"""

#Get Directory setting
"""
DICOM_DIR = "C:/Users/guigo/Documents/Databases/LIDC/LIDC-IDRI"
LIDC_IDRI_NODULE ="C:/Users/guigo/Documents/Databases/LIDC/"
MASK_DIR = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Mask/"
CLEAN_DIR_MASK = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Clean_Mask/"
META_DIR = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Meta/"
NODULE_DIR = "C:/Users/guigo/Documents/Databases/LIDC/Data_LIDC2/Nodule_2D/"

DICOM_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI"
LIDC_IDRI_NODULE ="/nas-ctm01/datasets/private/LUCAS/lidc/"
MASK_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Nodule_Mask/"
CLEAN_DIR_MASK = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Clean_Mask/"
META_DIR = "/nas-ctm01/datasets/private/LUCAS/lidc_nodule_masks/Meta/"


"""
DICOM_DIR = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms"
LIDC_IDRI_NODULE ="/nas-ctm01/datasets/public/MEDICAL/lidc-db"
MASK_DIR = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/lidc_nodule_masks/Nodule_Mask/"
CLEAN_DIR_MASK = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/lidc_nodule_masks/Clean_Mask/"
META_DIR = "/nas-ctm01/datasets/public/MEDICAL/lidc-db/lidc_nodule_masks/Meta/"



#Hyper Parameter setting for prepare dataset function
mask_threshold = 8
#Hyper Parameter setting for pylidc
confidence_level = 0.5
padding = 512

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, MASK_DIR,CLEAN_DIR_MASK,META_DIR,LIDC_IDRI_NODULE,mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.mask_path = MASK_DIR
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.nodule  = pd.read_csv(LIDC_IDRI_NODULE+'list3.2.csv')
        self.meta = pd.DataFrame(index=[],columns=['patient_id','dicom_path','slice_thickness','slice_spacing','pixel_spacing',
                                                   'nodule_no','slice_no','slice_no_inv','slice_mask_no',
                                                   'x_nod','y_nod','x_nod_est','y_nod_est',
                                                   'original_image','mask_image','malignancy','is_cancer','is_clean'])
        
        self.meta_central = pd.DataFrame(index=[],columns=['patient_id','dicom_path','slice_thickness','slice_spacing','pixel_spacing',
                                                           'nodule_no','slice_no','slice_no_inv','slice_mask_no',
                                                           'x_nod','y_nod','x_nod_est','y_nod_est',
                                                           'original_image','mask_image','malignancy','is_cancer','is_clean'])
        
        
    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','dicom_path','slice_thickness','slice_spacing','pixel_spacing',
                                         'nodule_no','slice_no','slice_no_inv','slice_mask_no',
                                         'x_nod','y_nod','x_nod_est','y_nod_est',
                                         'original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = pd.concat([self.meta, tmp.to_frame().T],ignore_index=True)
    def save_meta_central(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','dicom_path','slice_thickness','slice_spacing','pixel_spacing',
                                         'nodule_no','slice_no','slice_no_inv','slice_mask_no',
                                         'x_nod','y_nod','x_nod_est','y_nod_est',
                                         'original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta_central = pd.concat([self.meta_central, tmp.to_frame().T],ignore_index=True)
    def prepare_dataset(self):
        # This is to name each image and maskf
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)
        
        for patient in tqdm(self.IDRI_list):
            patient_i=int(patient.replace("LIDC-IDRI-",""))
            pid = patient #LIDC-IDRI-0001~
            nod_i=self.nodule.loc[self.nodule['case'] == patient_i]
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            dicom_path=scan.get_path_to_dicom_files()
            slice_thickness=scan.slice_thickness
            slice_spacing=scan.slice_spacing
            pixel_spacing=scan.pixel_spacing

            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_mask_dir = MASK_DIR / pid
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    
                    # Get the central slice of the computed bounding box.
                    k = int(0.5*(cbbox[2].stop - cbbox[2].start))
                    kk =np.shape(vol)[2]-(cbbox[2].start+k)
                    """
                    print(nod_i['slice no.'])
                    print(kk)
                    print(nod_i['slice no.'].isin([int(kk-2),int(kk-1),int(kk),int(kk+1),int(kk+2)]))
                    """
                    # match the nodules by the closer center slice
                    nod_i['diff']=np.abs(nod_i['slice no.']-kk)
                    no_slice=nod_i.nsmallest(1, 'diff')
                    
                    #no_slice=nod_i.loc[nod_i['slice no.'].isin([int(kk-2),int(kk-1),int(kk),int(kk+1),int(kk+2)])]
                    if (len(no_slice)>0):
                        nodule_x=no_slice['x loc.'].values[0]
                        nodule_y=no_slice['y loc.'].values[0]   
                    else:
                        nodule_x=-1
                        nodule_y=-1
                    
                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        s=cbbox[2].start+nodule_slice
                        ss=np.shape(vol)[2]-s
                        
                        y,x = np.where(mask[:,:,nodule_slice] > 0)
                        x_min=np.min(x)
                        y_min=np.min(y)
                        x_max=np.max(x)
                        y_max=np.max(y)
                        
                        nodule_x_estimated = int(x_min+(x_max-x_min)//2)
                        nodule_y_estimated = int(y_min+(y_max-y_min)//2)
                            
                        meta_list = [pid,dicom_path,slice_thickness,slice_spacing,pixel_spacing,
                                     nodule_idx,s,ss,prefix[nodule_slice],
                                     nodule_x, nodule_y, nodule_x_estimated,nodule_y_estimated,
                                     nodule_name,mask_name,malignancy,cancer_label,False]
                        self.save_meta(meta_list)
                        np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
                        
                        if (nodule_slice==k):
                            meta_list_central = [pid,dicom_path,slice_thickness,slice_spacing,pixel_spacing,
                                                 nodule_idx,s,ss,prefix[nodule_slice],
                                                 nodule_x, nodule_y, nodule_x_estimated,nodule_y_estimated,
                                                 nodule_name,mask_name,malignancy,cancer_label,False]
                            self.save_meta_central(meta_list_central)
                            
                    """
                    # Set up the plot.
                    fig,ax = plt.subplots(1,1,figsize=(5,5))
                    ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)

                    # Plot the annotation contours for the kth slice.
                    colors = ['r', 'g', 'b', 'y']
                    for j in range(len(masks)):
                        for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
                            label = "Annotation %d" % (j+1)
                            plt.plot(c[:,1], c[:,0], colors[j], label=label)
                    
                    # Plot the 50% consensus contour for the kth slice.
                    for c in find_contours(mask[:,:,k].astype(float), 0.5):
                        plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')

                    
                    plt.scatter(nodule_x, nodule_y, color='blue')
                    plt.scatter(nodule_x_estimated, nodule_y_estimated, color='green')
                    # plt.title("patient "+ patient+ " : "+ str(k)+"\n"+str(kk)+ " "+ str(no_slice['slice no.'].values[0]))
                    print(patient)
                    print(k,nodule_x_estimated, nodule_y_estimated)
                    """
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
                    if slice >50:
                        break
                    #CN= CleanNodule, CM = CleanMask
                    nodule_name = "{}/{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice])
                    mask_name = "{}/{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice])
                    meta_list = [pid,dicom_path,slice_thickness,slice_spacing,pixel_spacing,
                                         nodule_idx,s,ss,prefix[nodule_slice],
                                         -1, -1, -1,-1,
                                         nodule_name,mask_name,-1,False,True]
    
                    self.save_meta(meta_list)
                
                        
        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)
        self.meta_central.to_csv(self.meta_path+'meta_info_central.csv',index=False)

        return self.meta

# NI= Nodule Image, MA = Mask Original , CN = Clean Nodule , CM = Clean Mask
def is_nodule(row):
    if 'NI' in row:
        return True
    else:
        return False


def is_train(row,train,val,test):
    if row in train:
        return 'Train'
    elif row in val:
        return 'Validation'
    else:
        return 'Test'

def create_label_segmentation(meta, meta_central):
    patient_id = list(np.unique(meta['patient_id']))
    train_patient , test_patient = train_test_split(patient_id,test_size= 0.20)
    train_patient, val_patient = train_test_split(train_patient,test_size= 0.20)
    print(len(train_patient),len(val_patient),len(test_patient))
    
    meta['data_split']= meta['patient_id'].apply(lambda row : is_train(row,train_patient,val_patient,test_patient))
    meta_central['data_split']= meta_central['patient_id'].apply(lambda row : is_train(row,train_patient,val_patient,test_patient))

    return meta,meta_central

if __name__ == '__main__':
    
    print(DICOM_DIR,MASK_DIR,CLEAN_DIR_MASK,META_DIR,LIDC_IDRI_NODULE,mask_threshold,padding,confidence_level)
    
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test= MakeDataSet(LIDC_IDRI_list,MASK_DIR,CLEAN_DIR_MASK,META_DIR,LIDC_IDRI_NODULE,mask_threshold,padding,confidence_level)
    m=test.prepare_dataset()
    
    meta = pd.read_csv(META_DIR+'meta_info.csv')
    meta_central = pd.read_csv(META_DIR+'meta_info_central.csv')
        
    # comment if the classification is not binary
    meta_central=meta_central.loc[meta_central["is_cancer"] != "Ambiguous"]  
    meta=meta.loc[meta["is_cancer"] != "Ambiguous"]  
   
    meta['is_nodule']= meta['original_image'].apply(lambda row: is_nodule(row))
    meta_central['is_nodule']= meta_central['original_image'].apply(lambda row: is_nodule(row))

    # Lets separate Clean meta and meta data
    meta = meta[meta['is_nodule']==True]
    meta.reset_index(inplace=True)
    
    meta_central = meta_central[meta_central['is_nodule']==True]
    meta_central.reset_index(inplace=True)
    
    """
    meta_patient_id = list(np.unique(meta['patient_id']))
    # We need to train/test split independently for clean_meta, meta
    meta,meta_central = create_label_segmentation(meta,meta_central)
    """

    # Clean Meta only stores meta information of patients without nodules.
    meta.to_csv(META_DIR+'meta.csv')
    meta_central.to_csv(META_DIR+'meta_central.csv')
