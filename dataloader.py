import os
import torch
import torch.utils.data as data
import torchvision.transforms as transform
import numpy as np
from PIL import Image
import cv2

class AlzhDataset(data.Dataset):
    def __init__(self, type, root='/data/tm/alzh/New_dataset/train', transform=None):
        self.transform = transform
        if type == 'AD_CN':
            ## AD -> 0, CN -> 1
            image_file = os.path.join(root, 'images_AD_CN.txt')
            class_file = os.path.join(root, 'image_class_labels_AD_CN.txt')
        elif type == 'AD_MCI':
            ## AD -> 0, MCI -> 1
            image_file = os.path.join(root, 'images_AD_CN.txt')
            class_file = os.path.join(root, 'image_class_labels_AD_CN.txt')
        elif type == 'MCI_CN':
            ## MCI -> 0, CN -> 1
            image_file = os.path.join(root, 'images_AD_CN.txt')
            class_file = os.path.join(root, 'image_class_labels_AD_CN.txt')
        elif type == '3class':
            ## AD -> 0 MCI -> 1, CN -> 2
            image_file = os.path.join(root, 'images.txt')
            class_file = os.path.join(root, 'image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2class = self.list2dict(self.text_read(class_file))
        self.images = []
        self.labels = []
        for k in id2image.keys():
            image_path = os.path.join(root, id2image[k])
            self.images.append(image_path)
            self.labels.append(int(id2class[k]))


    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        label = self.labels[item]
        # canny = cv2.Canny(np.asarray(img), 50, 150, L2gradient=True) / 255.
        if self.transform is not None:
            img = self.transform(img)
        # img = torch.cat([img, torch.from_numpy(canny).unsqueeze(0)], dim=0).float()
        return img, label


if __name__ == '__main__':
    dataset = AlzhDataset()
    train_len = int(dataset.__len__() * 0.8)
    valid_len = dataset.__len__() - train_len
    train, valid = torch.utils.data.random_split(dataset, [train_len, valid_len])
    print(train.__len__(), valid.__len__())