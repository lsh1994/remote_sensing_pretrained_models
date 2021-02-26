from collections import OrderedDict
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import models
import pandas as pd
from data_loader import PlantDataset
from torch import optim
from torch import nn
from torchtoolbox.nn import LabelSmoothingLoss
from torchtoolbox.tools import mixup_data, mixup_criterion
import torchtoolbox.nn.loss as tloss
from model_finetune import *
from config import DefaultConfig as dc
import utils


class LitPlants(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net =Efficient(dc.backbone_name, dc.num_classes)

    def forward(self, x):
        y = self.net(x)
        return y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=dc.lr)
        scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        # scheduler0 ={
        #         'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=3,verbose=True),
        #         'monitor': 'val_acc',  # Default: val_loss
        #         'interval': 'epoch',
        #         'frequency': 1}
        # scheduler = utils.GradualWarmupScheduler(optimizer, 8, 10, scheduler1)

        return [optimizer], [scheduler1]

    # def prepare_data(self):
    #     data = pd.read_csv("data/data.csv",sep="\t")
    #     split = int(len(data) * 0.5)
    #     data = data.sample(frac=1, random_state=2020)  # 打乱
    #     data[:split].to_csv("data/train.csv", index=False, sep="\t")
    #     data[split:].to_csv("data/val.csv", index=False, sep="\t")

    ###############
    def train_dataloader(self):
        train_dataset = PlantDataset(
            "data/train.csv",
            transforms=dc.transforms_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=dc.batch_size,
            shuffle=True,
            num_workers=4
        )
        return train_loader

    def training_step(self, batch, batch_idx):  # mixup + labelsmooth+ circle_loss

        data, labels = batch

        # mixed_x, labels_a, labels_b, lam = mixup_data(data, labels, 0.2)
        # output = self(mixed_x)
        # loss = mixup_criterion(LabelSmoothingLoss(4, smoothing=0.1), output, labels_a, labels_b, lam)

        output = self(data)
        loss = LabelSmoothingLoss(dc.num_classes, smoothing=0.1)(output, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([output["loss"] for output in outputs]).mean()

        tqdm_dict = {"loss": loss}  # todo lr
        result = {'log': tqdm_dict, 'train_loss': loss, 'progress_bar': tqdm_dict}
        return result

    ##########
    def val_dataloader(self):
        val_dataset = PlantDataset(
            "data/val.csv",
            transforms=dc.transforms_val)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=dc.batch_size,
            shuffle=False,
            num_workers=4
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        acc = self.accuracy(output, target)

        output = OrderedDict({
            'val_acc': acc,
        })
        return output

    def validation_epoch_end(self, outputs):
        val_acc = torch.stack([output["val_acc"] for output in outputs]).mean()

        tqdm_dict = {"val_acc": val_acc}
        result = {'log': tqdm_dict, 'progress_bar': tqdm_dict}
        return result

    #########
    @classmethod
    def accuracy(cls, output, target):
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).float().sum() / batch_size
        return correct
