# ECO-paddle
该项目实现了动作识别方法**ECO** ，采用的数据集为 **UCF-101**，在 split-01 的测试集上达到  92.267% 的精度。

## **一、数据处理**
1、首先解压压缩包到 `data` 文件夹下。

```
unzip data/data48916/UCF-101.zip -d data/
```

2、对视频进行提帧，这里为了加速，采用了多线程的提帧方式，对所有视频提帧大概花费30分钟。

```
python avi2jpg.py
```

3、按照 split-01 的训练集\测试集划分方式，将每个视频的信息（视频名称、类别标签、所有帧的路径）保存到对应的文件夹下（train/val）；然后分别读取 train 和 val 文件夹下的视频信息，生成训练集、验证集标注文件。

```
python jpg2pkl.py
python data_list_generate.py
```

## **二、模型训练**

1、将 Kinetics 数据集上预训练过的 ECO-full torch 权重，转换为 paddle 权重。对应的 python 文件为 torch2paddle.py。

torch 预训练权重下载链接：[torch 权重](https://drive.google.com/open?id=1ATuN_KctsbFAbcNgWDlETZVsy2vhxZay)

2、训练模型，模型加载预训练权重，然后在 UCF-101 数据集上微调，训练的脚本文件为 run.sh:

```
sh run.sh
```

训练日志保存在了log目录下。

## **三、模型测试**

1、上述训练过程中保存了在验证集上精度最高的模型，下面单独使用该模型在测试集上进行测试：

```
sh test.sh
```

2、测试的日志保存在了log 目录下。

## 四、Reference

[ECO-pytorch](https://github.com/mzolfaghari/ECO-pytorch)