# runnable-PointPainting
This repo is used for successfully making PointPainting run. So some modifications on import etc. are made. Please feel free to check the original code of PointPainting if there is any error with this edited version.

This runnable PointPainting is tested on Ubuntu 18, with a very recent torch and CUDA 11.0. The trainning should be able to run without errors.

## Datasets config
I think you may need to point the dataset location in train.py to your own KITTI dataset source.
The folder tree should be:
-Kitti Dataset (only need to make sure this name is same as the file directory in the train.py)  
--trainning  
---calib  
---image_2  
---label_2  
---velodyne  

# PointPainting
An implementation of PointPainting (https://arxiv.org/abs/1911.10150), an image-lidar fusion algorithm for 3D object detection. Code is partly based on the [Pointpillars](https://github.com/nutonomy/second.pytorch) repo as well as [this](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) object detection repo. The algorithm involves performing a semantic segmentation of the image, projecting the pointcloud to the resulting segmentation map to give each lidar point its class score, and running the augmented pointcloud through a 2D object detector (SSD) in BEV. Some changes to the algorithm have been made including removal of orientation estimation, and changes to the feature extractor. Predictions are made for the Car class on the KITTI dataset.

![](prediction.png)
Above shows predicted (green) bounding boxes for an image and BEV map as well as ground truths (blue)

## Installation and Training steps
Clone the repo:
```
git clone https://github.com/rshilliday/painting.git
cd painting
pip install -r requirements.txt
```
Download and unzip the [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) (images, lidar, labels, calib)
Also download and save [weights](https://drive.google.com/file/d/1nqSDmTx97Y23j7L3Wca5hPudxXp57A7J/view?usp=sharing) for the semantic segmentation network

Train and evaluate the network:
```
python train.py
python eval.py
```
