[中文说明](./readme.zh.md)

# Why?

When we deal with remote sensing problems, we often use ImageNet pre-training weights to initialize the pre-network. ImageNet's natural images are quite different from remote sensing (scene) images, so the amount of data and the number of iterations are also higher. To this end, I trained some basic convolutional neural networks on some public data sets, hoping to better and faster transfer learning.

# How to use?

The code used with pytorch=1.4.0, python3.6.10.

You can download the trained weights through [Releases](https://github.com/lsh1994/remote_sensing_pretrained_models/releases) .  

In order to use the model, you can code as follows：
```python
import torch
from albumentations.pytorch import ToTensorV2
import model_finetune
import cv2
import albumentations as alb

# Model loading example
weights = torch.load(r"output/resnet34-epoch=9-val_acc=0.966.ckpt")["state_dict"] # Model weights
for k in list(weights.keys()):
    weights[str(k)[4:]]=weights.pop(k)

net = model_finetune.ResNet("resnet34",30)
net.load_state_dict(weights) # Load the weights
print(net)

```
Test a picture:
```python

labels_dict = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
     'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
     'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square',
     'Stadium', 'StorageTanks', 'Viaduct']
     
image = cv2.imread(r"AID/Viaduct/viaduct_256.jpg", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transforms_train = alb.Compose([
        alb.Resize(height=224, width=224, p=1),
        alb.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])
image = transforms_train(image=image)['image']
image = torch.unsqueeze(image,dim=0)

net.eval()
output = net(image)
output = torch.softmax(output,dim=1)
index =torch.argmax(output[0]).item()

print(output)
print(output[0,index].item(),labels_dict[index])
```

# Result

<!-- ## AID++ -->

<!-- ## BigEarthNet -->

## RSD46-WHU

[RSD46-WHU](https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU)

The data set has 117,000 256x256 pictures, 46 categories, 500\~3000 for each category, and a spatial resolution of 0.5\~2m. Training provided by the experiment (filtering non-pictures and duplicate files): Verification = 92110: 16810.

The network model experiment is as follows:      
| Network | Input size | Optimal number of iterations | Validation set accuracy |  publish weights |   
| :- | :-: | :-: | :-: | :-: |            
resnet34 | 256 | 19 | 0.921 | ✓   
densenet121 | 256 | 19 | 0.927 | ✓   
se_resnext50_32x4d | 224 | 19 | 0.930 | ✓    
efficientnet-b2 | 256 | 19 | 0.931 | ✓   

## AID
[AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification](https://captain-whu.github.io/AID/)

The data set has 10 000 600x600 pictures, 30 categories, 200\~400 for each category, and a spatial resolution of 0.5\~0.8m. Experimental split training: verification=8:2. Refer to "aid/eda.ipynb" under the folder for data analysis.

The network model experiment is as follows:  
| Network | Input size | Optimal number of iterations | Validation set accuracy |  publish weights | 
| :-: | :-: | :-: | :-: | :-: |       
| resnet34 | 224 | 9 | 0.966 | ✓ |  
resnet34 | 320 | 29 | 0.975 | ✗   
resnet34 | 600 | 26 | 0.981 | ✓    
densenet121 | 224 | 36 | 0.975 | ✓   
efficientnet-b2 | 224 | 27 | 0.979 | ✓   


Attachment: When the data is divided into 5:5, use resnet34, input size 224, and obtain the verification set accuracy of 0.959 at the 8th time.
