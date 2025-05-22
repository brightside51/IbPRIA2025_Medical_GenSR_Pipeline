# Imports
import numpy as np
import pandas as pd

# PyTorch Imports
from torch.utils import data



# Class: LIDCDataset
class LIDCDataset(data.Dataset):
    
    """
    Class describing a dataset for lung segmentation
    Attributes:
        - list_scans: List of CT scans in the dataset
        - data_path: path to directory containing scans and masks (preprocessing output)
        - mode: 2d will return slices
        - scan_size: size of CT-scans
        - n_classes: number of output class
    """

    def __init__(self, list_scans, list_masks,data_path,meta, n_classes=1):
        self.list_images = list_scans
        self.list_masks = list_masks
        self.data_path = data_path
        
        self.n_classes = n_classes
        self.meta=meta

        return


    # Method: __len__
    def __len__(self):
        return len(self.list_images)


    # Method: __getitem__
    def __getitem__(self, index):

        #try:
        #dtf_a = pd.read_csv("/nas-ctm01/datasets/private/LUCAS/LIDC-IDRI-Preprocessing/meta.csv")
        dft_a=pd.read_csv("/nas-ctm01/datasets/private/LUCAS/LIDC-IDRI-Preprocessing/meta_single_slice_v3.csv")
        stat_aug=list(self.meta["stat_aug"])[index]
        #scan_path = self.data_path+"Image_lung/"+self.list_images[index]+".npy"
        #mask_path = self.data_path+"Mask_roi32/"+self.list_masks[index]+".npy"
        #mask_path = self.data_path+"Mask_lung/"+self.list_masks[index]+".npy"

        scan_path = self.data_path+"Image_all/"+self.list_images[index]+".npy"
        #mask_path = self.data_path+"Mask_all/"+self.list_masks[index]+".npy"
        mask_path = self.data_path+"Mask_roi32_all/"+self.list_masks[index]+".npy"

        cls=dft_a[dft_a["original_image"]==self.list_images[index]]["is_cancer"]
        
        if cls.iloc[0]=="True":
            cls=1
        elif cls.iloc[0]=="False":
            cls=0


        ct_scan = np.load(scan_path)
        seg_mask = np.load(mask_path)
        # transform = A.OneOf([
        # A.HorizontalFlip()], p=1.0)
        # if stat_aug==1:
        #      ct_scan = transform(image=ct_scan)['image']
        #      seg_mask = transform(image=np.float32(seg_mask))['image']
        #      seg_mask=seg_mask.astype(np.bool)
        

        return ct_scan[np.newaxis, :, :], seg_mask[np.newaxis, :, :],cls
