import torch
from torch.utils.data import random_split
import torch.optim as optim

from torch.utils.data import  DataLoader
import os
import numpy as np
import argparse

import sys
sys.path.append(os.path.abspath('../..')) 
from EncoderModels.PointNet import PointNet, PointNetMedium, PointNetLarge, PointNetClassifier
from EncoderModels.dataSetClf import PointCloudDatasetClf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, train_loader, test_loader=None, model_name = None, 
          epochs=20, lr=1e-3, device=None, save_freq=None, save_path_base=None):
   
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_points, batch_labels in train_loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)  

            optimizer.zero_grad()
            outputs = model(batch_points)           
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_points.size(0)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        if epoch and (epoch + 1) % save_freq == 0:
            backbone_path =  save_path_base + f"/{model_name}/class_{epoch}.pth"
            full_model_path =  save_path_base + f"/{model_name}/full_model_class_{epoch}.pth"
            torch.save(model.backbone.state_dict(), backbone_path)
            torch.save(model.state_dict(), full_model_path)
            print(f"Backbone сохранён в {backbone_path}")
      
        if test_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_points, val_labels in test_loader:
                    val_points = val_points.to(device)
                    val_labels = val_labels.to(device)  

                    outputs = model(val_points)
                    loss = criterion(outputs, val_labels)

                    val_loss += loss.item() * val_points.size(0)
                    _, predicted = outputs.max(1)
                    val_total += val_labels.size(0)
                    val_correct += predicted.eq(val_labels).sum().item()

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
        net = PointNet()
    elif extractor_name == "mediumpn":
        net =  PointNetMedium()
    elif extractor_name == "largepn":
        net = PointNetLarge()
    save_freq = args.save_freq
    print('PointNetArchetecture')
    print(net)

    print("PointNet with classification head")
    pointNetCls = PointNetClassifier(net, 4)
    print(pointNetCls)

    print('Dataset Loading')
    dataSet = PointCloudDatasetClf('../../dexartEnv/assets/data', 4, {"bucket":0, 'faucet':1, "laptop":2, "toilet":3})

    train_ratio = 0.6
    train_size = int(len(dataSet) * train_ratio)
    test_size = len(dataSet) - train_size

    train_dataset, test_dataset = random_split(dataSet, [train_size, test_size])
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Train size:{train_size}, Test size: {test_size}")

    save_path = f"../../artifacts/Encoders"

    train(model=pointNetCls, train_loader=train_loader, test_loader=test_loader, model_name=extractor_name, epochs=epochs, save_path_base=save_path, save_freq=save_freq)