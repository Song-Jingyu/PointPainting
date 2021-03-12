import torch
import PIL
import os
import copy
import numpy as np
from torch import nn
from torchvision import transforms
import pickle
import torchvision
import math

class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, valid):
        self.root = root
        self.transforms = transforms
        self.mode = mode # 'training' or 'testing'
        if self.mode != 'training' and self.mode != 'testing':
            raise ValueError('mode must be "training" or "testing".')
        if valid == True and self.mode != 'training':
            raise ValueError('mode must be set to "training" if valid is set to True.')
        with open('file_nums_with_cars.pkl', 'rb') as f:
            self.file_nums_with_cars = pickle.load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        deeplab101 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        #Change out channels to 5 since that is num_categories for my KITTI-modified Cityscapes data
        deeplab101.classifier[4] = nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
        deeplab101.aux_classifier[4] = nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
        for p in deeplab101.backbone.parameters():
            p.requires_grad = False
        deeplab101.load_state_dict(torch.load('./deeplab_20epochs.pth'))
        deeplab101 = deeplab101.to(self.device)
        deeplab101.eval()
        self.deeplab101 = deeplab101

        train_size = len([s + '.png' for s in self.file_nums_with_cars]) # number of examples available for training
        train_limit = int(0.9 * train_size) # 90% of 'training' folder used for trainset, 10% for validset

        if self.mode == 'training':
            if valid == False: # trainset
                self.imgs = [s + '.png' for s in self.file_nums_with_cars][:train_limit]
                self.lidar = [s + '.bin' for s in self.file_nums_with_cars][:train_limit]
                self.calib = [s + '.txt' for s in self.file_nums_with_cars][:train_limit]
                self.labels = [s + '.txt' for s in self.file_nums_with_cars][:train_limit]

            else: # valset
                self.imgs = [s + '.png' for s in self.file_nums_with_cars][train_limit:]
                self.lidar = [s + '.bin' for s in self.file_nums_with_cars][train_limit:]
                self.calib = [s + '.txt' for s in self.file_nums_with_cars][train_limit:]
                self.labels = [s + '.txt' for s in self.file_nums_with_cars][train_limit:]

        else: # testset
            self.imgs = list(sorted(os.listdir(os.path.join(root, mode, "image_2"))))
            self.lidar = list(sorted(os.listdir(os.path.join(root, mode, "velodyne"))))
            self.calib = list(sorted(os.listdir(os.path.join(root, mode, "calib"))))

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.mode, "image_2", self.imgs[idx])
        lidar_path = os.path.join(self.root, self.mode, "velodyne", self.lidar[idx])
        calib_path = os.path.join(self.root, self.mode, "calib", self.calib[idx])

        with open(calib_path) as f:
            lines = f.readlines()
            for l in lines:
                l = l.split(':')[-1]

            R0_rect = np.eye(4)
            Tr_velo_to_cam = np.eye(4)

            P2 = np.array(lines[2].split(":")[-1].split(), dtype=np.float32).reshape((3,4))
            R0_rect[:3, :3] = np.array(lines[4].split(":")[-1].split(), dtype=np.float32).reshape((3,3)) # makes 4x4 matrix
            Tr_velo_to_cam[:3, :4] = np.array(lines[5].split(":")[-1].split(), dtype=np.float32).reshape((3,4)) # makes 4x4 matrix
            projection_mats = {'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam':Tr_velo_to_cam}

        img = PIL.Image.open(img_path).convert("RGB")
        pointcloud = np.fromfile(lidar_path, dtype=np.float32)
        pointcloud = pointcloud.reshape(-1,4)
        lidar_cam_coords = self.cam_to_lidar(pointcloud, projection_mats)
        class_scores = self.create_class_scores_mask(img)
        augmented_lidar_cam_coords = self.augment_lidar_class_scores(class_scores, lidar_cam_coords, projection_mats)

        if self.mode == 'training':
            label_path = os.path.join(self.root, self.mode, "label_2", self.labels[idx])
            with open(label_path) as f:
                labels = []
                lines = f.readlines()
                for l in lines:
                    if l.startswith('Car'):
                        bbox_2d = np.array(l.split()[4:8], dtype=np.float32)
                        dims_3d = np.array(l.split()[8:11], dtype=np.float32)
                        car_center_3d = np.array(l.split()[11:14], dtype=np.float32)
                        rotation_y = np.float32(l.split()[14])

                        # in pixel coords
                        left, top, right, bottom = bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3] 

                        # height(x-axis of camera [right/left]), width(y axis of camera [down/up]), length(z axis of camera[forward/back]) of car (meters)
                        h, w, l = dims_3d[0], dims_3d[1], dims_3d[2]
                        
                        # x,y,z in camera coords correspond to -y, -z, x in lidar coords (coords in meters)
                        x, y, z = car_center_3d[0], car_center_3d[1], car_center_3d[2]
                        
                        labels.append({'2d_bbox_img_coords': (left, top, right, bottom),'3d_bbox_dims_cam_coords': (h, w, l), '3d_car_center_cam_coords': (x, y, z), 'rotation_y': (rotation_y)})

            boxes, classes = self.create_boxes_and_labels(labels)
            boxes = boxes.to(self.device)
            classes = classes.to(self.device)
            return augmented_lidar_cam_coords, boxes, classes
        else:
            return augmented_lidar_cam_coords

    def __len__(self):
        return len(self.imgs)

    def cam_to_lidar(self, pointcloud, projection_mats):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo_to_cam'].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def create_class_scores_mask(self, img):
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        tensor_img = transform(img)
        tensor_img = tensor_img.unsqueeze(0).to(self.device)
        mask = self.deeplab101(tensor_img)
        mask = mask['out'] #ignore auxillary output
        _, preds = torch.max(mask, 1)
        class_scores = torch.where(preds==3, torch.ones(preds.shape).to(self.device), torch.zeros(preds.shape).to(self.device)) #convert preds to binary map (1 = car, else 0)
        class_scores = class_scores.squeeze()
        return class_scores

    def augment_lidar_class_scores(self, class_scores, lidar_cam_coords, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        reflectances = copy.deepcopy(lidar_cam_coords[:, -1])
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        points_projected_on_mask = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask = points_projected_on_mask.transpose()
        points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

        true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores.shape[1]) #x in img coords is cols of img
        true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores.shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

        points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
        lidar_cam_coords = torch.from_numpy(lidar_cam_coords[true_where_point_on_img])
        reflectances = reflectances[true_where_point_on_img]
        reflectances = torch.from_numpy(reflectances.reshape(-1, 1))
        points_projected_on_mask = np.floor(points_projected_on_mask).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask = torch.from_numpy(points_projected_on_mask[:, :2]) #drops homogenous coord 1 from every point, giving (N_pts, 2) int array
        
        #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        point_scores = class_scores[points_projected_on_mask[:, 1], points_projected_on_mask[:, 0]].reshape(-1, 1).double()
        augmented_lidar_cam_coords = torch.cat((lidar_cam_coords[:, :-1].to(self.device), reflectances.to(self.device), point_scores.to(self.device)), 1)

        return augmented_lidar_cam_coords

    def create_boxes_and_labels(self, labels, x_range=(-40, 40), z_range=(0, 80), pillar_resolution=0.16):
        """
        Creates inputs expected by Loss function. boxes is (n_obj, 4) tensor (xmin, ymin, xmax, ymax)
        """
        z_height = z_range[1] - z_range[0]
        x_width = x_range[1] - x_range[0]
        bev_rows = int(z_height/pillar_resolution)
        bev_cols = int(x_width/pillar_resolution)
        assert bev_rows == bev_cols #square bev img
        boxes = torch.empty(len(labels), 4)
        classes = torch.ones(len(labels)) # all are cars

        for i in range(len(labels)):
            rot = labels[i]['rotation_y']
            _, w, l = labels[i]['3d_bbox_dims_cam_coords']
            x, _, z = labels[i]['3d_car_center_cam_coords'] #x,z in cam coords are left/right & forward/back on bev

            if not (math.pi/4 < abs(rot) < (3*math.pi)/4): #if angle not in that range, car is facing left/right, so width is up/down
                w, l = l, w #make so width is always left/right on bev img
            
            xmin = math.floor((x - w/2 - x_range[0]) / pillar_resolution) #left 'pixel' coord of bbox (bev_img)
            xmax = math.floor((x + w/2 - x_range[0]) / pillar_resolution)# right
            ymin = math.floor((z_range[1] - (z + l/2)) / pillar_resolution) #top
            ymax = math.floor((z_range[1] - (z - l/2)) / pillar_resolution) #bottom

            boxes[i] = torch.tensor([xmin, ymin, xmax, ymax])

        boxes /= bev_rows # convert from bev pixel coords to bev fractional coords (0 to 1)
        return boxes, classes

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of lidar, lists of varying-size tensors of bounding boxes, labels
        """

        lidar = list()
        boxes = list()
        classes = list()

        for b in batch:
            lidar.append(b[0])
            boxes.append(b[1])
            classes.append(b[2])

        return lidar, boxes, classes
