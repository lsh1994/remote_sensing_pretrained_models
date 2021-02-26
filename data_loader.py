import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as alb
import cv2
import torchvision
import matplotlib.pyplot as plt
from config import DefaultConfig as dc
from skimage import io,color
import skimage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PlantDataset(Dataset):

    def __init__(self, df, transforms,is_test=False):
        self.df = pd.read_csv(df,sep="\t")
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # idx = 256 # 仅数据增强测试用
        image_src = self.df.loc[idx, 'path']
        image = io.imread(image_src)

        # print(image.shape)
        if len(image.shape)==2:
            image = color.gray2rgb(image)
        if image.shape[-1]==4:
            image = image[:,:,:3]

        # if image.shape[0]>image.shape[1]:
        #     image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE)

        transformed = self.transforms(image=image)
        image = transformed['image']

        if self.is_test:
            return image,image_src
        else:
            labels = dc.labels_dict.index(self.df.loc[idx, "label"])
            return image, labels

if __name__ == '__main__':

    train_dataset = PlantDataset(
        "data/train.csv",
        transforms=dc.transforms_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True
    )
    for x,y in train_loader:
        print(x.shape,y.shape,dc.labels_dict[y.item()])
        image = torchvision.utils.make_grid(x,4)
        image = np.transpose(image.numpy(),(1,2,0))
        image = image * dc.RGB_STD  + dc.RGB_MEAN

        cv2.imshow("",image[:,:,::-1])
        cv2.waitKey(0)

