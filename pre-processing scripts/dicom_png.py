#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import imageio
import numpy as np


# In[20]:


inDir = 'F:/LAB WORK/RIT/LAB/dataset_large/dicom_data/dicom_train/MCI'
outDir = 'F:/LAB WORK/RIT/LAB/dataset_large/png_data/train/MCI/'


# In[21]:


# folders = os.listdir(inDir)
# # print(folders)
# # print(len(folders))
# for folder in folders:
#     PATH = os.path.join(inDir, folder)
#     folder_2 = os.listdir(PATH)
#     for folder in folder_2:
#         PATH_2 = os.path.join(PATH, folder)
#         folder_3 = os.listdir(PATH_2)
#         for folder in folder_3:
#             PATH_3 = os.path.join(PATH_2, folder)
#             folder_4 = os.listdir(PATH_3)
#             print(folder_4)
#             break
#         break
#     break
        


# In[22]:


folders = os.listdir(inDir)
for folder in folders:
    PATH = os.path.join(inDir, folder)
    folder_2 = os.listdir(PATH)
    for folder in folder_2:
        PATH_2 = os.path.join(PATH, folder)
        folder_3 = os.listdir(PATH_2)
        for folder in folder_3:
            PATH_3 = os.path.join(PATH_2, folder)
            folder_4 = os.listdir(PATH_3)
            for folder in folder_4:
                PATH_4 = os.path.join(PATH_3, folder)
                images = [os.path.basename(x) for x in glob.glob(PATH_4 + '/*.dcm')]
                for f in images[10:35]:  
                    ds = dicom.read_file(PATH_4 + '/' + f) # read dicom image
                    img = ds.pixel_array # get image array
                    img = np.flip(img, axis=0)
                    imageio.imwrite(outDir + f.replace('.dcm','.png'), img)


# In[ ]:




