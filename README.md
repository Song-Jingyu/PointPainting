# PointPainting
This repository aims to build an open-source PointPainting package which is easy to understand, deploy and run! We refer to the idea in the [original paper](https://arxiv.org/abs/1911.10150) to implement this open-source framework to conduct the sequential 3D object detection.

## Update
We propose to support Kitti dataset first and utilize OpenPCDet as the LiDAR detection framework. We are expected to release the code to support Kitti and at least one semantic segmentation method to do painting by the end of April 2021.

## Table of Contents
- [PointPainting](#pointpainting)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
    - [Framework Overview](#framework-overview)
  - [Install](#install)
    - [OpenPCDet](#openpcdet)
    - [mmsegmentation](#mmsegmentation)
  - [How to Use](#how-to-use)
    - [Dataset Preparation](#dataset-preparation)
  - [Results](#results)
  - [Authors](#authors)

## Background
The PointPainting means to fuse the semantic segmentation results based on RGB images and add class scores to the raw LiDAR pointcloud to achieve higher accuracy than LiDAR-only approach.

### Framework Overview
The PointPainting architecture consists of three main stages: (1) image based semantics network, (2) fusion (painting), and (3) lidar based detector. In the first step, the images are passed through a semantic segmentation network obtaining pixelwise segmentation scores. In the second stage, the lidar points are projected into the segmentation mask and decorated with the scores obtained in the earlier step. Finally, a lidar based object detector can be used on this decorated (painted) point cloud to obtain 3D detections.
![](framework_overview.png)

## Install
To use this repo, first install these dependencies.
- [Pytorch](https://pytorch.org/), tested on Pytorch 1.6/1.7 with CUDA toolkit
- [OpenPCDet](#openpcdet)
- [mmsegmentation](#mmsegmentation)

### OpenPCDet

### mmsegmentation

## How to Use

### Dataset Preparation

## Results

## Authors