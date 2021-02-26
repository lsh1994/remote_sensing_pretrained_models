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

# 测试一张图片
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
labels_dict = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
     'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
     'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square',
     'Stadium', 'StorageTanks', 'Viaduct']
print(output)
print(output[0,index].item(),labels_dict[index])