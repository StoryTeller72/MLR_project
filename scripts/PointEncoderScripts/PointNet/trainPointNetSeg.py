import torch
from torch.utils.data import random_split
import torch.optim as optim

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import numpy as np
import argparse


import sys
sys.path.append(os.path.abspath('../..')) 
from EncoderModels.PointNet  import PointNet, PointNetMedium, PointNetLarge, PointNetSegmentationHead
from scripts.PointEncoderScripts.DataSetSeg import PointCloudDatasetSetSeg

def train_segmentation(model, train_loader, test_loader=None, epochs=10, lr=1e-4, save_freq=None, save_path_base=None,  model_name = None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # метки по точкам: (B, N)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_points = 0

        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)  # (B, N)

            optimizer.zero_grad()
            outputs = model(points)     # (B, N, num_classes)
            loss = criterion(outputs.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * points.size(0)
            _, preds = outputs.max(2)
            correct += (preds == labels).sum().item()
            total_points += labels.numel()
        if epoch and (epoch + 1) % save_freq == 0:
            backbone_path =  save_path_base + f"/{model_name}/seg_{epoch}.pth"
            full_model_path =  save_path_base + f"/{model_name}/full_model_seg_{epoch}.pth"
            torch.save(model.backbone.state_dict(), backbone_path)
            torch.save(model.state_dict(), full_model_path)
            print(f"Backbone сохранён в {backbone_path}")

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total_points
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        if test_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for points, labels in test_loader:
                    points = points.to(device)
                    labels = labels.to(device)
                    outputs = model(points)
                    loss = criterion(outputs.transpose(1, 2), labels)
                    val_loss += loss.item() * points.size(0)
                    _, preds = outputs.max(2)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()

            val_loss /= len(test_loader.dataset)
            val_acc = val_correct / val_total
            print(f"  Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=1)
    args = parser.parse_args()


    extractor_name = args.extractor_name
    epochs = args.epochs
    if extractor_name == "smallpn":
        backBone = PointNet()
    elif extractor_name == "mediumpn":
        backBone =  PointNetMedium()
    elif extractor_name == "largepn":
        backBone = PointNetLarge()
    save_freq = args.save_freq
    print('PointNetArchetecture')
    print(backBone)

    print("PointNet with segmantation head")
    pointNetSeg = PointNetSegmentationHead(backBone, 4)
    print(pointNetSeg)

    print('Dataset Loading')
    dataSet = PointCloudDatasetSetSeg('../../dexartEnv/assets/data')

    train_ratio = 0.7
    train_size = int(len(dataSet) * train_ratio)
    test_size = len(dataSet) - train_size

    train_dataset, test_dataset = random_split(dataSet, [train_size, test_size])
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Train size:{train_size}, Test size: {test_size}")

    save_path = f"../../artifacts/Encoders"

    train_segmentation(pointNetSeg, train_loader, test_loader, model_name=extractor_name, epochs=epochs, save_path_base=save_path, save_freq=save_freq)

