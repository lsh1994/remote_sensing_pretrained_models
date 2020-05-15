# 起因

当我们处理遥感问题时候，经常使用ImageNet预训练权重对前置网络初始化。ImageNet的自然图像与遥感（场景）图像有较大区别，所以数据量、迭代次数等要求的也更高。为此，我在一些公开数据集上训练了一些基础卷积神经网络，希望能够更好更快的迁移学习。

# 使用

代码基于 pytorch=1.4.0, python3.6.10。

你可以通过 [Releases](https://github.com/lsh1994/remote_sensing_pretrained_models/releases) 下载训练好的权重.  

为了使用模型，你可以编码如下：
```python
import torch
from albumentations.pytorch import ToTensorV2
import model_finetune
import cv2
import albumentations as alb

# 模型加载样例
weights = torch.load(r"output/resnet34-epoch=9-val_acc=0.966.ckpt")["state_dict"] # 模型权重
for k in list(weights.keys()):
    weights[str(k)[4:]]=weights.pop(k)

net = model_finetune.ResNet("resnet34",30)
net.load_state_dict(weights) # 加载权重字典
print(net)

```
测试一张图片:
```python
# 测试一张图片
labels_dict = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
     'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
     'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square',
     'Stadium', 'StorageTanks', 'Viaduct']
     
image = cv2.imread(r"D:/Game_lsh/Gloabel_data/AID/Viaduct/viaduct_256.jpg", cv2.IMREAD_COLOR)
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

# 结果

## AID
[AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification](https://captain-whu.github.io/AID/)

数据集有10 000张 600x600 图片，30类别，每类200\~400，空间分辨率0.5\~0.8m。实验拆分训练：验证=8：2。参考文件夹下 "aid/eda.ipynb" 获取数据分析。

网络模型实验如下： 
网络 | 输入尺寸  | 最优迭代次数 | 验证集精度 | 是否发布权重
:- | :-: | :-: | :-: | :-:   
resnet34 | 224 | 9 | 0.966 | ✓
resnet34 | 320 | 29 | 0.975 | ✗
resnet34 | 600 | 26 | 0.981 | ✓
densenet121 | 224 | 36 | 0.975 | ✓
efficientnet-b2 | 224 | 27 | 0.979 | ✓

附：当数据划分为5：5时，使用resnet34,输入尺寸224，在第8次获得验证集精度0.959。

## RSD46-WHU

[RSD46-WHU](https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU)

数据集有117 000张 256x256 图片，46类别，每类500\~3000，空间分辨率0.5\~2m。实验提供的训练：验证 = 92110：16810 = 。

网络模型实验如下： 
网络 | 输入尺寸  | 最优迭代次数 | 验证集精度 | 是否发布权重
:- | :-: | :-: | :-: | :-:   
resnet34 | 256 | 19 | 0.921 | ✓
densenet121 | 256 | 19 | 0.927 | ✓
se_resnext50_32x4d | 224 | 19 | 0.930 | ✓
efficientnet-b2 | 256 | - | - | ✓

<!-- ## AID++ -->


<!-- ## BigEarthNet -->
