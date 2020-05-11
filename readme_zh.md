中文说明更新延迟.

# 为什么

当我们处理遥感问题时候，经常使用ImageNet预训练权重对前置网络初始化。ImageNet的自然图像与遥感（场景）图像有较大区别，所以数据量、迭代次数等要求的也更高。为此，我在一些公开数据集上训练了一些基础卷积神经网络，希望能够更好更快的迁移学习。

# 怎么做

训练代码基于 keras=2.2.4，tensorflow=1.12.0，python3.6。

所用的公开数据有:  
## AID
[AID: A Benchmark Dataset for Performance Evaluation of 
Aerial Scene Classification](http://captain.whu.edu.cn/WUDA-RSImg/aid.html)

数据集有10000张 600x600 图片，30类别。  
![](./others/aid-dataset.png)  
![](./others/class_count.png)

- 已经使用了VGG16、Resnet50训练。


# 效果如何

训练脚本比较简陋，预想取得更好的测试集精度或泛化能力，可以使用更好更多的数据增强、正则化技术和训练策略。

数据 | 网络 | 迭代次数 | batch_size |训练集精度  | 测试集精度
:-: | :-: | :-: | :-: | :-: | :-:
aid | vgg16 | 200 | 256 | 0.9540 | 0.8812-0.8953
aid | resnet50 | 200 | 128 | 0.9951 | 0.8625-0.9282

# 将来计划

更新更多。