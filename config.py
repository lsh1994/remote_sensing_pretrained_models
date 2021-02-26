import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2


class DefaultConfig:
    labels_dict = ['Airplane', 'Airport', 'Artificial dense forest land', 'Artificial sparse forest land',
                   'Bare land', 'Basketball court', 'Blue structured factory building', 'Building', 'Construction site',
                   'Cross river bridge', 'Crossroads', 'Dense tall building', 'Dock', 'Fish pond', 'Footbridge',
                   'Graff', 'Grassland', 'Low scattered building', 'Lrregular farmland', 'Medium density scattered building',
                   'Medium density structured building', 'Natural dense forest land', 'Natural sparse forest land', 'Oiltank',
                   'Overpass', 'Parking lot', 'Plasticgreenhouse', 'Playground', 'Railway', 'Red structured factory building',
                   'Refinery', 'Regular farmland', 'Scattered blue roof factory building', 'Scattered red roof factory building',
                   'Sewage plant-type-one', 'Sewage plant-type-two', 'Ship', 'Solar power station', 'Sparse residential area',
                   'Square', 'Steelsmelter', 'Storage land', 'Tennis court', 'Thermal power plant', 'Vegetable plot', 'Water']


    H,W = 256,256
    RGB_MEAN = np.array((0.485, 0.456, 0.406))
    RGB_STD = np.array((0.229, 0.224, 0.225))
    lr = 1e-3

    ## train
    batch_size = 32 # gpu存放
    accumulate_size = 128
    backbone_name = "efficientnet-b2"
    num_classes = 46
    pre_weight = None  # None
    transforms_val = alb.Compose([
        alb.Resize(height=H, width=W, p=1.0),
        alb.Normalize(p=1.0, mean=RGB_MEAN, std=RGB_STD),
        ToTensorV2(p=1.0),
    ])
    transforms_train = alb.Compose([
        alb.Resize(height=H, width=W, p=1),
        alb.RandomResizedCrop(height=H, width=W, p=1.0, scale=(0.8, 1.2)),

        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        # alb.ShiftScaleRotate(rotate_limit=45, p=0.5),
        alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        alb.CoarseDropout(p=0.5),
        alb.Normalize(p=1.0, mean=RGB_MEAN, std=RGB_STD),
        ToTensorV2(p=1.0),
    ])

    ## test
    # best_weights = r"output\lightning_logs\resnet34-epoch=96-val_acc=0.972.ckpt"
    # test_batch_size = 8