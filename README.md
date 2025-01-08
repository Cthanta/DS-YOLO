# Despeckling Representation for Data-Efficient SAR Ship Detection (DS-YOLO)

This repository is the official PyTorch implementation of DS-YOLO to SAR Ship Detection 

 ---

> Deep learning techniques are extensively applied to synthetic aperture radar (SAR) ship detection tasks. Nonetheless, the limited availability of labeled SAR images impedes the neural network's ability to learn and extract robust objects features from SAR ship images. To alleviate the dependence on large datasets, this letter introduces a despeckle-based representation learning approach for SAR ship detection, named Despeckling Ship Detection YOLO (DS-YOLO). The DS-YOLO model integrates a shared feature extractor, a detection head, and a despeckling head, facilitating the concurrent performance of SAR image despeckling and ship detection. The model effectively reduces the potential for neural network overfitting by conducting joint learning of detection and despeckling processes. As a result, DS-YOLO is particularly well-suited for SAR ship detection tasks with constrained training data. Comprehensive experiments indicate that DS-YOLO substantially surpasses the performance of the conventional detection model, particularly in scenarios where labeled data is severely restricted.
><p align="center">
  > <img height="400" src="./illustrations/MLMC.png">
</p>

## Requirements
- pip install -r requirements.txt 


## Train DS-YOLO
```bash
python train_ssl.py
```
Please **Take care to change the value from the hyperparameter `lamda`** before training.

---

## Val DS-YOLO
```bash
python val_ssl.py
```
---

## Detect DS-YOLO
```bash
python detect_ssl.py
```
---


## Dataset Preparation 

LS-SSDD datasets can be downloaded [here](https://radars.ac.cn/web/data/getData?newsColumnId=6b535674-3ce6-43cc-a725-9723d9c7492c).

SSDD datasets can be downloaded [here](https://gitcode.com/gh_mirrors/of/Official-SSDD/blob/main/README.md?utm_source=csdn_github_accelerator&isLogin=1).

## Note
The pre-training weights are in **"\DS-YOLO\runs\train\SSL"**
Dataset configuration documents ".yaml" are in **"\DS-YOLO\data"**
Model configuration documents ".yaml" are in **"\DS-YOLO\models\SSL"**

## Acknowledgement

 The codes are based on [YOLOv5](https://github.com/ultralytics/yolov5). Thanks for their great works.