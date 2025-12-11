import torch
import torch.nn as nn


class PointNet(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        super(PointNet, self).__init__()
        print(f'PointNet')
        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # pc = x[0].cpu().detach().numpy()
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetMedium(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        super(PointNetMedium, self).__init__()

        print(f'PointNetMedium')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetLarge(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        super(PointNetLarge, self).__init__()

        print(f'PointNetLarge')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x

# PointNet++

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




class PointNet2Lite(nn.Module):
    def __init__(self):
        super().__init__()
        print("PointNet2")
        self.sa1 = SA_Lite(npoint=256, k=16, in_channel=3, mlp=[32, 64])


    def forward(self, xyz):
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points) 
        out_put = l1_points
        out_put = torch.max(out_put, dim=1)[0]
        return out_put



class PointNet2Medium(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # SA слои
        self.sa1 = SA_Lite(npoint=256, k=16, in_channel=3, mlp=[32, 64])
        self.sa2 = SA_Lite(npoint=128, k=16, in_channel=64+3, mlp=[64, 128])

       

    def forward(self, xyz):
        points = None

        # SA слои
        l1_xyz, l1_points = self.sa1(xyz, points)          # [B, 256, 3], [B, 64, 256]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # [B, 128, 3], [B, 128, 128]

        # Надо дописать.