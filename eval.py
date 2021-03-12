from utils import *
from datasets import KittiDataset
import torch
import numpy as np
import os
import pickle

checkpoint = './pointpillars.pth'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
root = './'
results = root + 'results'
if not os.path.exists(results):
    os.makedirs(results)
batch_size = 4
workers = 0
bev_len = 500
ymax_cam, ymin_cam = 1.21, 2.91
h = ymin_cam - ymax_cam
z = (ymax - ymin) / 2
theta = np.pi/2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cam_to_img = np.array([[ 7.25995079e+02,  9.75088160e+00,  6.04164953e+02, 4.48572807e+01],
       [-5.84187313e+00,  7.22248154e+02,  1.69760557e+02, 2.16379106e-01],
       [ 7.40252715e-03,  4.35161404e-03,  9.99963105e-01, 2.74588400e-03]])

with open('file_nums_with_cars.pkl', 'rb') as f:
    file_nums_with_cars = pickle.load(f)

model.eval()

test_dataset = KittiDataset(root=root, mode="testing")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    with torch.no_grad():
        for i, augmented_lidar_cam_coords in enumerate(test_loader):
            augmented_lidar_cam_coords = augmented_lidar_cam_coords.to(device)  
            predicted_locs, predicted_scores = model(augmented_lidar_cam_coords)
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.2, max_overlap=0.45,
                                                                                       top_k=10)
            label = [l.to(device) for l in labels]
            score = [s.to(device) for s in scores]
            boxes = [b.to(device) for b in boxes]*bev_len
            xmin, ymin, xmax, ymax = boxes[0], boxes[1], boxes[2], boxes[3]
            x = (xmax - xmin) / 2
            y = (ymax - ymin) / 2
            w, l = xmax - xmin, ymax - ymin
            if w > l: w, l, theta = l, w, 0
            cam_points = np.array([[xmin, ymin_cam, z, 1], [xmax, ymax_cam, z, 1]]).T
            img_points = cam_to_img.dot(cam_points).T
            f_name = file_nums_with_cars[i] + '.txt'
            with f = open(f_name,"w+"):
                f.write(f"Car 0 0 0 {img_points[0]}, {img_points[1]}, {img_points[2]} {img_points[3]} {h} {w} {l} {x} {y} {z} {theta} {score}")


evaluate(test_loader, model)
