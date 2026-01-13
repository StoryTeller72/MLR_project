import torch
import torch.nn as nn


def knn(xyz, query_xyz, k):
    dist = torch.cdist(query_xyz, xyz)  # [B, S, N]
    idx = dist.topk(k, dim=-1, largest=False)[1]  # [B, S, k]
    return idx

def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return xyz[batch_indices.unsqueeze(-1), centroids]

def group_points(points, idx):
    # points: [B, N, C], idx: [B, S, k] -> [B, C, S, k]
    B, N, C = points.shape
    S, K = idx.shape[1], idx.shape[2]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1)
    grouped = points[batch_indices, idx]  # [B, S, k, C]
    return grouped.permute(0, 3, 1, 2)    # [B, C, S, k]


class SA_Lite(nn.Module):
    def __init__(self, npoint, k, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.k = k
        layers = []
        last_c = in_channel
        for c in mlp:
            layers.append(nn.Conv2d(last_c, c, 1))
            layers.append(nn.ReLU())
            last_c = c
        self.mlp = nn.Sequential(*layers)
        self.out_channel = mlp[-1]

    def forward(self, xyz, points):
        new_xyz = farthest_point_sample(xyz, self.npoint)  # [B, S, 3]
        idx = knn(xyz, new_xyz, self.k)                     # [B, S, k]
        grouped_xyz = group_points(xyz, idx)               # [B, 3, S, k]

        if points is not None:
            grouped_pts = group_points(points.permute(0, 2, 1), idx)  # [B, C, S, k]
            grouped = torch.cat([grouped_xyz, grouped_pts], dim=1)
        else:
            grouped = grouped_xyz

        new_points = self.mlp(grouped).max(dim=-1)[0]  # [B, C_out, S]
        return new_xyz, new_points


class FP_Lite(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_c = in_channel
        for c in mlp:
            layers.append(nn.Conv1d(last_c, c, 1))
            layers.append(nn.ReLU())
            last_c = c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        # xyz1: [B, N, 3] (fine), xyz2: [B, S, 3] (coarse)
        # points1: [B, C1, N] или None, points2: [B, C2, S]
        dist = torch.cdist(xyz1, xyz2) + 1e-8  # [B, N, S]
        inv = 1.0 / dist
        weight = inv / torch.sum(inv, dim=-1, keepdim=True)  # [B, N, S]
        interpolated = torch.matmul(weight, points2.permute(0, 2, 1)).permute(0, 2, 1)  # [B, C2, N]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=1)  # [B, C1+C2, N]
        else:
            new_points = interpolated
        return self.mlp(new_points)


class PointNet2LiteBacbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = SA_Lite(npoint=256, k=16, in_channel=3, mlp=[32, 64])

    def forward(self, xyz):
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points)  # [B, 256, 3], [B, 64, 256]
        return l1_xyz, l1_points


    

class PointNet2LiteClassHead(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()

        self.backbone = backbone

       
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, xyz):

        l1_xyz, l1_points = self.backbone(xyz)  # [B, 64, 256]

        
        global_feat = torch.max(l1_points, dim=2)[0]  # [B, 64]

        out = self.classifier(global_feat)  # [B, num_classes]
        return out

class PointNet2LiteSegmentation(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

       
        self.fp1 = FP_Lite(in_channel=64+3, mlp=[64, 64])

       
        self.head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, xyz):
        l1_xyz, l1_points = self.backbone(xyz)  # [B, 256, 3], [B, 64, 256]

       
        l0_up = self.fp1(xyz, l1_xyz, xyz.permute(0,2,1), l1_points)  # [B, 64, N]

       
        out = self.head(l0_up)  # [B, num_classes, N]
        return out.permute(0, 2, 1)  # [B, N, num_classes]