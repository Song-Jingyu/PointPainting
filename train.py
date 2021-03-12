import torch
import copy
from model import SSD, MultiBoxLoss
from dataset import KittiDataset
from torch.utils.data import DataLoader
import time

def train_model(model, dataloaders, criterion, optimizer, num_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                print(f"Starting train phase for epoch {epoch}")
                model.train()
            else:
                print(f"Starting validation phase for epoch {epoch}")
                model.eval()

            batch_num = 0
            running_loss = 0.0
            running_corrects = 0
            iteration_num = 0

            for augmented_lidar_cam_coords, boxes, classes in dataloaders[phase]:
                iteration_num += 1
                bat_size = len(augmented_lidar_cam_coords)
                num_samples_so_far = iteration_num * bat_size
                if num_samples_so_far % 100 == 0:
                    print("Samples processed for this epoch:", num_samples_so_far,'/',len(dataloaders[phase])*bat_size)
                    print("Average Loss so far this epoch is:", running_loss/(num_samples_so_far-bat_size))
                if batch_num % 100 == 0:
                    print(f'phase is {phase} and batch is {batch_num}.')
                batch_num += 1

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    predicted_locs, predicted_scores, _ = model(augmented_lidar_cam_coords)
                    loss = criterion(predicted_locs, predicted_scores, boxes, classes)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * bat_size
    return model #, val_acc_history
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ssd = SSD(resnet_type=34, n_classes=2).to(device)
trainset = KittiDataset(root="/media/jingyu/LoveBB/KITTI Dataset", mode="training", valid=False)
valset = KittiDataset(root="/media/jingyu/LoveBB/KITTI Dataset", mode="training", valid=True)

datasets = {'train': trainset, 'val': valset}
dataloaders_dict = {x: DataLoader(datasets[x], batch_size=2, shuffle=True, collate_fn=datasets[x].collate_fn, num_workers=0, drop_last=True) for x in ['train', 'val']}

optimizer_ft = torch.optim.SGD(ssd.parameters(), lr=0.0001, momentum=0.9)
criterion = MultiBoxLoss(priors_cxcy=ssd.priors_cxcy).to(device)

ssd = train_model(ssd, dataloaders_dict, criterion, optimizer_ft, num_epochs=10)
torch.save(ssd.state_dict(), './pointpillars.pth')
