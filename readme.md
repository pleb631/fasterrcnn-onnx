# faster-rcnn 原生onnx导出和推理

## 说明

本仓库存放了mmdet中Faster R-CNN  ONNX格式的导出代码，以及配套的推理代码。

建立该仓库的原因在于，目前主流开源项目中存在以下两点问题：

1. 推理代码通常嵌入在 pytorch 的训练框架中，与训练逻辑高度耦合，结构复杂，不利于直观地理解算法的执行流程;

2. 部分项目仅提供 C++ 版本的推理实现，这对于想要快速学习和验证算法原理，门槛较高;

3. 例如mmdet等仓库在导出二阶段模型时使用了自定义算子，进一步增加了模型的导出与使用复杂度，不利于原型开发

## 配置环境

该环境主要用于配合[mmdet](https://github.com/open-mmlab/mmdetection/tree/v2.28.2)进行模型导出。Python 版本选择 3.8，是因为 mmdet 所依赖的自定义算子位于 mmcv 中，而 mmcv 在更高版本下没有提供对应的预编译 wheel，需要手动编译，操作较为繁琐，因此采用了较为稳定的 3.8 版本环境。

```bash
# python3.8.20
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

代码会把`backbone+RPN`导出`model1.onnx`, `ROIHead`导出`model2.onnx`.并进行精度验证，如果精度误差都小于1e-4,会输出15个`True`

## 模型推理

确保ckpt文件夹有导出好的onnx文件，确保精度验证通过。
运行

```python
python demo.py
```
