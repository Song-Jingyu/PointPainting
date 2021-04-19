#!/bin/bash

wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth

conda install pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .
mkdir checkpoints
mv ../deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth ./checkpoints
cd ..
