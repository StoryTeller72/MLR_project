import torch
import torch.nn as nn


class PointNet(nn.Module):  
    def __init__(self, point_channel=3, output_dim=256):
        super(PointNet, self).__init__()

        print(f'PointNetSmall')

        in_channel = point_channel
        mlp_out_dim = output_dim
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
        # Local
        x = self.local_mlp(x)
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        return x


class PointNetClassifier(nn.Module):
    def __init__(self, backbone, num_classes=4):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)   
        out = self.fc(features)       
        return out



class PointNetSegBackbone(nn.Module):
    def __init__(self, point_channel=3, output_dim=256):
        super().__init__()

        print(f'PointNetSmall')

        in_channel = point_channel
        mlp_out_dim = output_dim
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
        local_feat = self.local_mlp(x)          # [B, N, C]
        global_feat = torch.max(local_feat, dim=1)[0]  # [B, C]
        return local_feat, global_feat


class PointNetSeg(nn.Module):
    def __init__(
        self,
        backbone: PointNet,
        feat_dim=256,
        num_classes=10
    ):
        super().__init__()

        self.backbone = backbone

        self.seg_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.seg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, N, 3]
        return: [B, N, num_classes]
        """

        B, N, _ = x.shape

        # --- Backbone ---
        local_feat, global_feat = self.backbone(x)
        # local_feat:  [B, N, C]
        # global_feat: [B, C]

        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)

        feat = torch.cat([local_feat, global_feat], dim=-1)  # [B, N, 2C]
        logits = self.seg_head(feat)

        return logits


