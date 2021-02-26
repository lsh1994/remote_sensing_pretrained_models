from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import albumentations as alb
from torch.utils.data import DataLoader
from data_loader import PlantDataset
from model import LitPlants
import tqdm
import torch
import cv2
import glob
import imutils
from config import DefaultConfig as dc
import glob
import pandas as pd
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def result_error():

    ################## 错误图像可视化
    model = LitPlants.load_from_checkpoint(r"output/resnet34-epoch=29-val_acc=0.975.ckpt").cuda()
    model.eval()
    test_dataset = PlantDataset(
        "data/val.csv",
        transforms=dc.transforms_val, is_test=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False
    )

    s = 0
    for k, (x, y) in enumerate(test_loader):
        pred = torch.softmax(model(x.cuda()), dim=1)
        for img, i, j in zip(x, y, pred):  # 图像，真实标签，预测值
            img = np.transpose(img.numpy(), (1, 2, 0))[:, :, ::-1]
            img = imutils.resize(img * dc.RGB_STD + dc.RGB_MEAN, 540)
            if torch.argmax(j).item() != i:
                print(f"批量：{k},预测值：{dc.labels_dict[torch.argmax(j).item()]}, 真实值：{dc.labels_dict[i]}，预测错误")
                s += 1
                cv2.imshow("", img)
                cv2.waitKey(0)
            else:
                print(f"{k},预测正确")

    print(s, len(test_dataset),1-s/len(test_dataset))

def result_combine():
    ## 结果投票
    path = glob.glob("output/tta-*.csv")

    r = pd.read_csv(path[0]).to_numpy()
    oh = r[:,1:]
    for i in path[1:]:
        oh += pd.read_csv(i).to_numpy()[:,1:]
    oh /= len(path)
    res = np.concatenate((r[:,0].reshape(-1,1),oh),axis=1)
    pd.DataFrame(res,columns=['image_id','healthy', 'multiple_diseases', 'rust', 'scab']).to_csv("output/combine.csv",index=False)

def create_samples(dir,savep):
    data = []
    labels = sorted(os.listdir(dir))
    print(labels)
    for i in labels:
        for j in os.listdir(os.path.join(dir, i)):
            if "副本" in j or j[-3:] not in ["png"]:
                print(i,j)
                break
            data.append([f"{dir}/{i}/{j}", i])
    data = pd.DataFrame(data=data, columns=["path", "label"])
    print(data.head())
    # data.to_csv(savep, index=False, sep="\t")

def view_all():
    data = pd.read_csv("data/train.csv",sep="\t")
    for i,d in data.iterrows():
        print(d["path"],i)
        # img = cv2.imdecode(np.fromfile(d["path"],dtype=np.uint8),-1)
        img = cv2.imread(d["path"],cv2.IMREAD_COLOR)
        print(img.shape, d["label"])
        # img = img[:,:,:3]
        print()
    print(data.head())

if __name__ == '__main__':
    # result_error()
    # result_combine()
    # create_samples("D:/Game_lsh/LSRSCI/train","data/train.csv")
    # create_samples("D:/Game_lsh/LSRSCI/val", "data/val.csv")
    view_all()
    pass
