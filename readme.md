# faster-rcnn 原生onnx导出和推理

## 说明

本仓库存放了Faster R-CNN的导出onnx的代码，以及配套的推理代码。目标是打通mmdet训练+导出+推理的流程，代码可复用，并且要锁死版本确保可复现。

建立该仓库的原因在于，目前主流开源项目中有以下三点问题：

1. 推理代码通常嵌入在 pytorch 的训练框架中，与训练逻辑高度耦合，结构复杂，不利于直观地理解算法的执行流程;

2. 部分项目仅提供 C++ 版本的推理实现，这对于想要快速学习和验证算法原理，门槛较高;

3. 例如mmdet等仓库在导出二阶段模型时使用了自定义算子，进一步增加了模型的导出与使用复杂度，不利于原型开发。

![流程图](asset/1.jpg)

## 配置环境

该环境主要用于配合[mmdet](https://github.com/open-mmlab/mmdetection/tree/v2.28.2)进行模型导出。Python 版本选择 3.8，是因为 mmdet 所依赖的自定义算子位于 mmcv 中，而 mmcv 在更高版本下没有提供对应的预编译 wheel，需要手动编译，操作较为繁琐，因此采用了较为稳定的 3.8 版本环境。

```bash
# python3.8
# 需严格按照下面顺序执行，重要的包需要锁死版本
pip install -r requirement.txt

mim install mmcv-full==1.7.2

pip install mmdet==2.28.2
git clone https://github.com/open-mmlab/mmdetection.git --branch v2.28.2 --depth 1

```

## 模型导出

首先需要下载相关权重文件，mmdet提供了相关[权重](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)，下载放在ckpt文件夹
之后运行代码

```python
python export.py
```

代码会把`backbone+RPN`导出`model1.onnx`, `ROIHead`导出`model2.onnx`.

### `model1.onnx`

输入是图片，格式为`[1,3,H,W]`

输出一共有15个

1. 第0-4个输出是每一层特征的分类分数，格式为`[1,3,H/s,W/s]`,`s`为每一层的`stride`,`3`对应三种预设框；
2. 第5-9个输出是每一层特征的回归框，格式为`[1,12,H/s,W/s]`，`12`是对应三个预设框的4个坐标回归值；
3. 第10-14个输出是每一层特征，格式为`[1,256,H/s,W/s]`,`256`是每一层特征统一降维至256通道，用来为后面ROIAlign做准备。

### `model2.onnx`

输入为ROIAlign之后的区域框特征，格式为`[B,256,7,7]`，`B`为第一阶段筛选的框的个数

输出有两个

1. 第0个输出为该框的分类分数，格式为`[B,81]`，其中有80类是前景类，第81类是背景类
2. 第1个输出为该框的回归值，格式为`[B,320]`，`320`对应的是每一个前景类的回归值，这个和yolo有很大区别，yolo是回归一组值，而这个是为每一个类都回归一组值。

### 输出对齐

在导出模型后，会自动对pytorch版本和onnx版本的模型的输出进行对齐，如果两个输出的误差均小于1e-4,会输出17个`True`。如果出现`False`,则视作精度有极大损失，导出失败。

## 模型推理

确保ckpt文件夹有导出好的onnx文件，输出对齐通过。
运行

```python
python demo.py
```

![流程图](asset/out.jpg)
