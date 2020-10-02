import torch.utils.data as data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as datasets
from PIL import Image
import os
import skimage.io as io
from PIL import Image


class MultiTaskDataset(Dataset):
    """My dataset for regression task."""

    def __init__(self, mr_csv, ct_csv, mr_root_dir, ct_root_dir, transform=None):
        self.transform = transform
        
        self.mr_csv_pth = mr_csv
        self.mr_csv_data = pd.read_csv(self.mr_csv_pth)[['Patient', 'age_at_initial_pathologic']]
        
        self.ct_csv_pth = ct_csv
        self.ct_csv_data = pd.read_csv(self.ct_csv_pth)[['Patient Number', 'Age\n(years)']]
        
        self.mr_imgs_pth = mr_root_dir
        self.mr_imgs = [os.path.join(self.mr_imgs_pth, img) for img in os.listdir(self.mr_imgs_pth)]
        self.mr_number = len(self.mr_imgs)
        
        self.ct_imgs_pth = ct_root_dir
        self.ct_imgs = [os.path.join(self.ct_imgs_pth, img) for img in os.listdir(self.ct_imgs_pth)]
        self.ct_number = len(self.ct_imgs)

    def __len__(self):
        return max(self.mr_number, self.ct_number)
    
    def __getitem__(self, idx):
        idx_ct = idx % self.ct_number
        idx_mr = idx % self.mr_number
        
        
        # CT part
        img_pth_ct = self.ct_imgs[idx_ct]
        img_name_ct = img_pth_ct.split('/')[-1]
        # Get image
        image_ct = Image.open(img_pth_ct).convert('RGB')
        if self.transform:
            image_ct = self.transform(image_ct)
        # Get label
        patient_id_ct = int(img_name_ct.split('-')[0])
        label_ct = int((self.ct_csv_data).loc[self.ct_csv_data['Patient Number'] == patient_id_ct, 'Age\n(years)'])
        
        
        # MR part
        img_pth_mr = self.mr_imgs[idx_mr]
        img_name_mr = img_pth_mr.split('/')[-1]
        # Get image
        image_mr = Image.open(img_pth_mr).convert('RGB')
        if self.transform:
            image_mr = self.transform(image_mr)
        # Get label
        patient_id_mr = img_name_mr[:12]
        label_mr = int((self.mr_csv_data).loc[self.mr_csv_data['Patient'] == patient_id_mr, 'age_at_initial_pathologic'])
        
        
        return image_ct, label_ct, image_mr, label_mr
    
        
        
        
        
        

class MultiTaskPETDataset(Dataset):
    """My dataset for regression task."""

    def __init__(self, mr_csv, ct_csv, pet_root_dir, ct_root_dir, transform=None):
        self.transform = transform
        
#         self.mr_csv_pth = mr_csv
#         self.mr_csv_data = pd.read_csv(self.mr_csv_pth)[['Patient', 'age_at_initial_pathologic']]

        self.pet_csv_pth = './pet.csv'
        self.pet_csv_data = pd.read_csv(self.pet_csv_pth)
        

        
        self.ct_csv_pth = ct_csv
        self.ct_csv_data = pd.read_csv(self.ct_csv_pth)[['Patient Number', 'Age\n(years)']]
        
#         self.mr_imgs_pth = mr_root_dir
#         self.mr_imgs = [os.path.join(self.mr_imgs_pth, img) for img in os.listdir(self.mr_imgs_pth)]
#         self.mr_number = len(self.mr_imgs)
    
        self.pet_imgs_pth = pet_root_dir
        self.pet_imgs = [os.path.join(self.pet_imgs_pth, img) for img in os.listdir(self.pet_imgs_pth)]
        self.pet_number = len(self.pet_imgs)

        self.ct_imgs_pth = ct_root_dir
        self.ct_imgs = [os.path.join(self.ct_imgs_pth, img) for img in os.listdir(self.ct_imgs_pth)]
        self.ct_number = len(self.ct_imgs)

    def __len__(self):
        return max(self.pet_number, self.ct_number)
    
    def __getitem__(self, idx):
        idx_ct = idx % self.ct_number
        idx_pet = idx % self.pet_number
        
        
        # CT part
        img_pth_ct = self.ct_imgs[idx_ct]
        img_name_ct = img_pth_ct.split('/')[-1]
        # Get image
        image_ct = Image.open(img_pth_ct).convert('RGB')
        if self.transform:
            image_ct = self.transform(image_ct)
        # Get label
        patient_id_ct = int(img_name_ct.split('-')[0])
        label_ct = int((self.ct_csv_data).loc[self.ct_csv_data['Patient Number'] == patient_id_ct, 'Age\n(years)'])
        
        
#         # MR part
#         img_pth_mr = self.mr_imgs[idx_mr]
#         img_name_mr = img_pth_mr.split('/')[-1]
#         # Get image
#         image_mr = Image.open(img_pth_mr).convert('RGB')
#         if self.transform:
#             image_mr = self.transform(image_mr)
#         # Get label
#         patient_id_mr = img_name_mr[:12]
#         label_mr = int((self.mr_csv_data).loc[self.mr_csv_data['Patient'] == patient_id_mr, 'age_at_initial_pathologic'])
    
        img_pth = self.pet_imgs[idx_pet]
        img_name = img_pth.split('/')[-1]
        # Get image
        # image = Image.fromarray(io.imread(img_pth), mode='RGB')
        image = Image.open(img_pth).convert('RGB')
        if self.transform:
            image_pet = self.transform(image)
        # Get label
        patient_id = '-'.join(img_name.split('-')[:3])
        label_pet = int((self.pet_csv_data).loc[self.pet_csv_data['Patient #'] == patient_id, 'Age'])
        
        
        return image_ct, label_ct, image_pet, label_pet