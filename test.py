import torch
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import tqdm
from data_loader import PlantDataset
from model import LitPlants
import albumentations as alb
import pandas as pd
import time
import numpy as np
from config import DefaultConfig as dc

##################
model = LitPlants.load_from_checkpoint(dc.best_weights).cuda()
model.eval()
test_dataset = PlantDataset(
    "data/test.csv",
    transforms=dc.transforms_val, is_test=True)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=dc.test_batch_size,
    shuffle=False
)

tm = str(time.strftime('%m%d%H%M'))
for epoch in range(1):
    print(f"tta:{epoch}")
    result = []
    for img, image_ids in tqdm.tqdm(test_loader):
        pred = torch.softmax(model(img.cuda()), dim=1)
        for i, j in zip(image_ids, pred.cpu()):
            result.append([i] + [t for t in j.detach().numpy()])
    result= pd.DataFrame(result,columns=['image_id','healthy', 'multiple_diseases', 'rust', 'scab'])
    result.to_csv(f"output/tta-{epoch}_{tm}.csv",index=False)