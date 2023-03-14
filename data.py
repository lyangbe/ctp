from torch.utils.data import Dataset
import pydicom
import os
import numpy as np
from torch import cat

#data loading for 1 person
class Dataset3D(Dataset):
    def __init__(self,ctp_image_path,tmax_image_path,cbv_image_path,cbf_image_path):
        super().__init__()
        self.ctp_image_path = ctp_image_path
        self.tmax_image_path = tmax_image_path
        self.cbv_image_path = cbv_image_path
        self.cbf_image_path = cbf_image_path
        self.data_prepare()

    def __len__(self):
        return self.num_of_position

    #return a t*image size array and its tmax, cbv, cbf
    def __getitem__(self,idx):
        return self.data_3d[idx],self.tmax[idx],self.cbv[idx],self.cbf[idx]
    

    # divide the images into groups by their position
    def data_prepare(self):
        ctp_image = [pydicom.dcmread(self.ctp_image_path + '/' + s) for s in sorted(os.listdir(self.ctp_image_path))]
        tmax = [pydicom.dcmread(self.tmax_image_path + '/' + s) for s in sorted(os.listdir(self.tmax_image_path))]
        cbv = [pydicom.dcmread(self.cbv_image_path + '/' + s) for s in sorted(os.listdir(self.cbv_image_path))]
        cbf = [pydicom.dcmread(self.cbf_image_path + '/' + s) for s in sorted(os.listdir(self.cbf_image_path))]
        for img in tmax:
            img = img.pixel_array
        for img in cbv:
            img = img.pixel_array
        for img in cbf:
            img = img.pixel_array
        self.tmax = tmax
        self.cbv = cbv
        self.cbf = cbf
        
        self.data_3d = []
        # the number of pictures in one input group, = timepoints in the same position
        self.num_of_position = int(ctp_image[0].SliceThickness) # how many different positions
        self.timepoints_in_one_position = len(ctp_image) // self.group_size # how many different timepoints for each position
        for p in range(self.num_of_position):
            concat_data = ctp_image[p*self.timepoints_in_one_position]
            for t in range(1,self.timepoints_in_one_position):
                concat_data = cat(concat_data,ctp_image[t * self.num_of_position + p].pixel_array)
            self.data_3d.append(concat_data)