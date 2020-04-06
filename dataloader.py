import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class Image_Loader(Dataset):
    def __init__(self, root_path='./data_train.csv', image_size=[192, 192], transforms_data=True):
        
        self.data_path = pd.read_csv(root_path)
        self.image_size = image_size
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, item):

        # load 
        image_path = os.path.join(self.data_path.iloc[item, 0])
        image = Image.open(image_path)
        
        # label to train cross_entropy loss
        label_cross = self.data_path.iloc[item, 1]

        #normalize the label by dividing it to 100
        label_cross = label_cross/100

        if self.transforms_data == True:
            data_transform = self.transform(True, True, True)
            image = data_transform(image)

        return image, torch.from_numpy(np.array(label_cross, dtype=np.float32))

    def transform(self, flip, resize, totensor):
        options = []

        if flip:
            options.append(transforms.RandomHorizontalFlip())
        if resize:
            options.append(transforms.Resize(self.image_size))
        if totensor:
            options.append(transforms.ToTensor())
        
        transform = transforms.Compose(options)

        return transform