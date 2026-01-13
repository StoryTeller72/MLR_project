import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from EncoderModels.PointNet2 import PointNet2LiteBacbone, PointNet2LiteClassHead 
import torch
import torch.optim as optim

from torch.utils.data import  DataLoader
import os
import numpy as np
sys.path.append(os.path.abspath('')) 

import argparse

from scripts.PointEncoderScripts.utils import ClsDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def train(model, device,  train_loader, val_loader=None, model_name = None, 
          epochs=20, lr=1e-3, save_freq=None, save_path_base=None):
    logsAccuracy = []
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_points, batch_labels in train_loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)  
            batch_labels = batch_labels - 1
            optimizer.zero_grad()
            outputs = model(batch_points)           
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_points.size(0)
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

            batch_points = batch_points.to('cpu')
            batch_labels = batch_labels.to('cpu')  

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        logsAccuracy.append(epoch_acc)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        if epoch and (epoch + 1) % save_freq == 0  and save_path_base:
            backbone_path =  save_path_base + f"/class_{epoch}.pth"
            torch.save(model.backbone.state_dict(), backbone_path)
            print(f"Backbone сохранён в {backbone_path}")
      
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            i = 0
            with torch.no_grad():
                for val_points, val_labels in val_loader:
                    val_points = val_points.to(device)
                    val_labels = val_labels.to(device)  
                    val_labels = val_labels - 1
                    outputs = model(val_points)
                    loss = criterion(outputs, val_labels)

                    val_loss += loss.item() * val_points.size(0)
                    _, predicted = outputs.max(1)
                    val_total += val_labels.size(0)
                    val_correct += predicted.eq(val_labels).sum().item()
                    val_points = val_points.to('cpu')
                    val_labels = val_labels.to('cpu')  

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total
            print(f"  Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")
    return logsAccuracy

def test(model, device, test_loader, model_name):
    print('Test results')
    total = 0
    correct = 0
    for batch_points, batch_labels in test_loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)
            batch_labels = batch_labels - 1  
            outputs = model(batch_points)           

            total += batch_labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()

       
    accuracy = correct / total

    print(f"Test Acc: {accuracy:.4f}")



if __name__ == '__main__':
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..', '..'))
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'artifacts', 'dataCls')
    TRAIN_PATH = os.path.join(DATA_ROOT, 'train.npz')
    VAL_PATH   = os.path.join(DATA_ROOT, 'val.npz')
    TEST_PATH  = os.path.join(DATA_ROOT, 'test.npz')

    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'artifacts/Encoders/PointNet2class')
    LOGS_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'res.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default="pn2")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=1)
    args = parser.parse_args()


    extractor_name = args.extractor_name
    epochs = args.epochs
    bacbone = PointNet2LiteBacbone(4)
    model = PointNet2LiteClassHead(4, bacbone)
    save_freq = args.save_freq
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("PointNet with classification head")
    print(model)




    print('Dataset Loading')
    dataset_train = ClsDataset(TRAIN_PATH)
    dataset_val = ClsDataset(VAL_PATH)
    dataset_test = ClsDataset(TEST_PATH)
    batch_size = 512
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    print(f"DEVICE: {device},Train size:{len(dataset_train)}, Validation size: {len(dataset_val)}, Test size: {len(dataset_test)}")

    logs = train( model , device, loader_train, loader_val, 'PN2_class', 5, save_freq=5, save_path_base=MODEL_SAVE_PATH)

    with open(LOGS_SAVE_PATH, 'w') as file:
        file.write('\n'.join(map(str, logs)))
    test(model, device,  loader_test, 'PnC2lass')
    model.to('cpu')