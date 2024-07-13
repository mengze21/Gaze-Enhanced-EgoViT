# Gaze Enhanced EgoViT
This is the code for master thesis *Gaze-based Transformer for Improving Action Recognition in Egocentric Videos*


## Introduction
This repo is pytorch implementaion of our Gaze Enhanced EgoViT.

A pretrained HOD module is used in this project, which is aviliable at [hand-object detector](https://github.com/ddshan/hand_object_detector).

## Prerequisites
The installition and usage of HOD please follow the [Readme of HOD](https://github.com/ddshan/hand_object_detector/blob/master/README.md)

Create a conda env called gazeEgoViT with python 3.9 and install the newer pytorch and dependences:
* PyTorch==2.2.0
* cuda==12.1
* torchversion==0.17.0
* opencv-python==4.10.0.82
* numpy==2.2.1
* pandas==2.2.1
* tqdm==4.66.4

## Train
### Data preparation
The dataset stores in folder gaze_dataset, which includes the video clips and gaze tracking data. Detail see [gaze_readme.md](gaze_dataset/gaze_readme.md)

Use [GazeExtractor.py](gaze_preprocessing/GazeExtractor.py) to generate the gaze point coordinates for every video clips

Use [run.py](run.py) to save the sampled frams in videos and run HOD modeule.

Run [GazeFea_extractor_224](GazeFeatures/GazeFea_extractor_224.py) to extract the gaze features from frames. 

Use [combin_data.py](GazeFeatures/combin_data.py) and [prepocess_data.py](transformer/preprocess_data.py) to generate the trainin/testing dataset.

Use [train.py](transformer/train.py) to train the modle or use [resume_training.py](transformer/resume_training.py) to train the model with resume function.

## Test
Evaluate the modle with [test.py](transformer/test.py)

## About the modle and dataset

The gaze enhanced EgoViT and its variant model are in file EgoViT_swinb.py.

[myDataset.py](transformer/myDataset.py) and [myDataset_v2.py](transformer/myDataset_v2.py) are the custom dadatset of gaze version 1 and gaze version 2