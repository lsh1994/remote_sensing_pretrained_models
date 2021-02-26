import warnings
warnings.filterwarnings('ignore')
from model import *
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from albumentations.pytorch import ToTensorV2
import albumentations as alb
import numpy as np
from config import DefaultConfig as dc

if __name__ == '__main__':

    pre_weight = dc.pre_weight

    trainer = pl.Trainer(
        default_save_path="output/",
        gpus=1,
        accumulate_grad_batches=dc.accumulate_size//dc.batch_size,
        amp_level="o1",  # GPU半精度
        max_epochs=200,
        checkpoint_callback=pl.callbacks.model_checkpoint.ModelCheckpoint(
            filepath='output/'+dc.backbone_name+'-{epoch}-{val_acc:.3f}',
            monitor='val_acc', verbose=True,save_top_k=1,save_weights_only=True),
        # early_stop_callback=pl.callbacks.early_stopping.EarlyStopping(monitor='val_acc', patience=5,verbose=True),
    )

    if pre_weight:
        net = LitPlants.load_from_checkpoint(pre_weight)
        trainer.fit(net)
    else:
        net = LitPlants()
        trainer.fit(net)
