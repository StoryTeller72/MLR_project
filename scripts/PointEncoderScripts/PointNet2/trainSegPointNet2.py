import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from EncoderModels import PointNet2
import torch
from torch.utils.data import random_split
import torch.optim as optim

from torch.utils.data import  DataLoader
import os
import numpy as np
import argparse
import sys
sys.path.append(os.path.abspath('')) 
from scripts.PointEncoderScripts.utils import SemSegDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def train(model, train_loader,config):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if config['cat'] == 'bucket':
        print('Using Weighted CrossEntropy loss')
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([3., 1., 1., 1.]).to(device))
    else:
        print('Using CrossEntropy loss')
        criterion = nn.CrossEntropyLoss()
    for epoch in range(config['num_epochs']):
        tic = time.time()
        model.train()
        for i, (points, labels) in enumerate(train_loader):
            points = points.float().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            
            outputs = model(points)
            # outputs: [B, N, 4], labels: [B, N]
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            if i % config['log_step'] == 0:
                print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
               

   

def save(model, epoch, config):
    print( f"{config['log_dir']}/{epoch}.pth")
    torch.save(model.sa1.state_dict(), f"{config['log_dir']}/{epoch}.pth")





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='pn2', help='model architecture')
    parser.add_argument('--cat', type=str, default='bucket', help='category to train')
    parser.add_argument('--run', type=str, default='0', help='run id')
    parser.add_argument('--use_img', action='store_true', help='use image', default=False)
    parser.add_argument('--eval', type=str, default=None, help='eval model name e.g. pn_100.pth')
    args = parser.parse_args()


    arch = args.arch
    cat = args.cat
    run = args.run
    use_img = args.use_img
    point_channel = 3
    num_epochs = 100
    config = {
        'num_epochs': num_epochs,
        'log_step': 10,
        'val_step': 1,
        'log_dir': '/home/rustam/ProjectMy/artifacts/PointNet2',
        'arch': arch,
        'lr': 1e-3,
        'classes': 4,
        'save_step': 20,
        'cat': cat,
    }


    train_dataset = SemSegDataset(split='train', point_channel=point_channel, use_img=use_img, root_dir=f'/home/rustam/ProjectMy/artifacts/DataSeg/{cat}')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    
    model = PointNet2.PointNet2Lite(4)
    train(model, train_loader, config)
    save(model,num_epochs, config)


    # eval()
