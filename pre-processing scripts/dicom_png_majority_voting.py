#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import imageio
import numpy as np


# In[9]:


CATEGORIES = ['AD', 'CN', 'MCI']
inDir = 'F:/LAB WORK/RIT/LAB/dataset_large/dicom_data/dicom_test/'
outDir = 'F:/LAB WORK/RIT/LAB/dataset_large/png_data/test_scanwise/'


# In[10]:


for categ in CATEGORIES:
    folders = os.listdir(inDir + categ + '/')
    print(len(folders))


# In[11]:


categ = CATEGORIES[0]
i = 1
folders = os.listdir(inDir + categ + '/')
print(folders)


# In[12]:


# categ = CATEGORIES[0]
# i = 1

# folders = os.listdir(inDir + categ + '/')
# # print(folders)
# # print(len(folders))
# for folder in folders:
#     PATH = os.path.join(inDir, categ, folder)
#     folder_2 = os.listdir(PATH)
#     break
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


# In[16]:


categ = CATEGORIES[2]
i = 1
#os.mkdir(outDir + categ + '/' + f'{i}')
folders = os.listdir(inDir + categ + '/')
for folder in folders:
    PATH = os.path.join(inDir, categ, folder)
    folder_2 = os.listdir(PATH)
    for folder in folder_2:
        PATH_2 = os.path.join(PATH, folder)
        folder_3 = os.listdir(PATH_2)
        for folder in folder_3:
            PATH_3 = os.path.join(PATH_2, folder)
            folder_4 = os.listdir(PATH_3)
            for folder in folder_4:
                PATH_4 = os.path.join(PATH_3, folder)
                os.mkdir(outDir + categ + '/' + f'{i}')
                out_path = outDir + categ + '/' + f'{i}'
                images = [os.path.basename(x) for x in glob.glob(PATH_4 + '/*.dcm')]
                for f in images[10:35]:  
                    ds = dicom.read_file(PATH_4 + '/' + f) # read dicom image
                    img = ds.pixel_array # get image array
                    img = np.flip(img, axis=0)
                    imageio.imwrite(out_path + '/' + f.replace('.dcm','.png'), img)
                i += 1


# In[ ]:




