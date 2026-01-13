import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from EncoderModels.PointNet import PointNetSegBackbone, PointNetSeg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
import sys
sys.path.append(os.path.abspath('')) 
from scripts.PointEncoderScripts.utils import SemSegDataset
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def confusion_matrix(preds, labels, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def compute_metrics(cm):
    num_classes = cm.shape[0]

    iou_per_class = []
    acc_per_class = []

    for c in range(num_classes):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP

        denom = TP + FP + FN
        iou = TP.float() / denom.float() if denom > 0 else torch.tensor(0.)
        acc = TP.float() / (TP + FN).float() if (TP + FN) > 0 else torch.tensor(0.)

        iou_per_class.append(iou)
        acc_per_class.append(acc)

    mIoU = torch.mean(torch.stack(iou_per_class))
    mAcc = torch.mean(torch.stack(acc_per_class))
    OA = torch.trace(cm).float() / cm.sum().float()

    return {
        "IoU": iou_per_class,
        "Acc": acc_per_class,
        "mIoU": mIoU.item(),
        "mAcc": mAcc.item(),
        "OA": OA.item()
    }

def save_logs_json(logs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(logs, f, indent=4)

def train(model, train_loader, config):
    logs = []

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step'], 
        gamma=config['lr_gamma']      
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_classes = 4  

    if config['cat'] == 'bucket':
        print('Using Weighted CrossEntropy loss')
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([3., 1., 1., 1.]).to(device)
        )
    else:
        print('Using CrossEntropy loss')
        criterion = nn.CrossEntropyLoss()

    for epoch in range(config['num_epochs']):
        model.train()

        # confusion matrix за эпоху
        epoch_cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

        for i, (points, labels) in enumerate(train_loader):
            points = points.float().to(device)    # [B, N, 3]
            labels = labels.long().to(device)     # [B, N]

            optimizer.zero_grad()

            outputs = model(points)               # [B, N, 4]
            outputs = outputs.permute(0, 2, 1)    # [B, 4, N]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)         # [B, N]

            epoch_cm += confusion_matrix(
                preds.view(-1).cpu(),
                labels.view(-1).cpu(),
                num_classes
            )

            if i % config['log_step'] == 0:
                print(
                    f"Epoch [{epoch + 1}/{config['num_epochs']}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )
        if epoch % config['save_step'] ==0 :
            save(model,epoch, config)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"  LR   : {current_lr:.6e}")

        metrics = compute_metrics(epoch_cm)

        logs.append({
            'epoch': epoch + 1,
            'loss': loss.item(),
            'OA': metrics['OA'],
            'mAcc': metrics['mAcc'],
            'mIoU': metrics['mIoU']
        })

        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  OA   : {metrics['OA']:.4f}")
        print(f"  mAcc : {metrics['mAcc']:.4f}")
        print(f"  mIoU : {metrics['mIoU']:.4f}")

        print("  IoU per class:")
        for c, iou in enumerate(metrics['IoU']):
            print(f"    Class {c}: {iou:.4f}")

        print("-" * 40)
        save_logs_json(logs, '/home/rustam/ProjectMy/artifacts/Encoders/PointNetSeg/logs.json')
    return logs

               

        

def save(model, epoch, config):
    print( f"{config['log_dir']}/{epoch}.pth")
    torch.save(model.backbone.state_dict(), f"{config['log_dir']}/{epoch}.pth")





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='pn', help='model architecture')
    parser.add_argument('--cat', type=str, default='bucket', help='category to train')
    parser.add_argument('--run', type=str, default='0', help='run id')
    parser.add_argument('--use_img', action='store_true', help='use image', default=False)
    args = parser.parse_args()


    arch = args.arch
    cat = args.cat
    run = args.run
    use_img = args.use_img
    point_channel = 3
    num_epochs = 50
    config = {
        'num_epochs': num_epochs,
        'log_step': 10,
        'val_step': 1,
        'log_dir': '/home/rustam/ProjectMy/artifacts/Encoders/PointNetSeg',
        'arch': arch,
        'lr': 1e-3,
        'lr_step': 10,     # каждые 10 эпох
        'lr_gamma': 0.5,   # lr *= 0.5
        'classes': 4,
        'save_step': 5,
        'cat': cat,
    }


    train_dataset = SemSegDataset(split='train', point_channel=point_channel, use_img=use_img, root_dir=f'/home/rustam/ProjectMy/artifacts/DataSeg/{cat}')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    backbone = PointNetSegBackbone()
    model = PointNetSeg(backbone, num_classes=4)
    logs = train(model, train_loader, config)
    torch.save(model.state_dict(), '/home/rustam/ProjectMy/artifacts/Encoders/PointNetSeg/fullModel.pth')
    # save_logs_json(logs, '/home/rustam/ProjectMy/artifacts/Encoders/PointNetSeg/logs.json')


