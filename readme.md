[中文说明](./readme_zh.md)  
# Why?

When we deal with remote sensing problems, ImageNet pre-training weights are often used to initialize the pre-network. The natural image is quite different from remote sensing (scene) image, so the amount of data and the number of iterations are higher needed. So, I trained some basic convolutional neural networks on some public datasets, hoping to migrate better and faster.

# How?

This code used keras=2.2.4, tensorflow=1.12.0, python3.6.  
You can download weights in [Releases](https://github.com/lsh1994/remote_sensing_pretrained_models/releases).  
You can code as:
```python
import keras
from keras.layers import Dense,Dropout

nb_classes = 45 

path = "rs_pretrain_model_aid-vgg16-notop.h5" # weights path

mod = keras.applications.VGG16(include_top=False,
        weights=None,
        input_shape=(224,224,3),
        pooling="avg") # base model

assert isinstance(mod,keras.models.Model)

mod.load_weights(path) # load pretrained weights 

x = mod.output # add layers to classify or others
x = Dense(1024,activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes,activation="softmax")(x)
sqeue = keras.models.Model(inputs=mod.input,outputs=x)
```

Public data used is:  
## AID
[AID: A Benchmark Dataset for Performance Evaluation of 
Aerial Scene Classification](http://captain.whu.edu.cn/WUDA-RSImg/aid.html)

The data set has 10,000 600x600 images and 30 categories.  
![](./others/aid-dataset.png)  
Dataset includes 8500 train images,1500 test images. All distribution:    
![](./others/class_count_aid.png)


# Result?

Training scripts are relatively rudimentary and similar.If you want to get better test set precision or generalization ability, you can use better and more data enhancement, regularization techniques and training strategies. All data size is scaled to 224x224. AID uses Data-Augmentation(Augmentation costs more time), while RScup2019 limits less.

Data | Network | Iterations | batch_size | Max Training Set Accuracy | Max Test Set Accuracy  
:-: | :-: | :-: | :-: | :-:  | :-: 
AID | VGG16 | 200 | 256 | 0.9540 | 0.8953  
AID | ResNet50 | 200 | 128 | 0.9951 | 0.9282  
AID | DenseNet121 | 50 | 128 | 0.9879 | 0.9355    
RScup2019 | DenseNet121 | 20 | 128 | - | -   


# Next?

Update more. I am not sure.