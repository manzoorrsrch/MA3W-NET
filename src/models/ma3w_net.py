import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = [x2.size(d) - x1.size(d) for d in range(2, 5)]
        x1 = F.pad(
            x1,
            [diff[2]//2, diff[2]-diff[2]//2,
             diff[1]//2, diff[1]-diff[1]//2,
             diff[0]//2, diff[0]-diff[0]//2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BoundaryHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch // 2, 3, padding=1),
            nn.InstanceNorm3d(ch // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch // 2, 1, 1),
        )
    def forward(self, x):
        return self.conv(x)

class M3ANeuroSeg(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, dims=(32,64,128,256)):
        super().__init__()
        d1, d2, d3, d4 = dims

        self.inc   = ConvBlock(in_ch, d1)
        self.down1 = Down(d1, d2)
        self.down2 = Down(d2, d3)
        self.down3 = Down(d3, d4)

        self.up1 = Up(d4, d3)
        self.up2 = Up(d3, d2)
        self.up3 = Up(d2, d1)

        self.outc    = nn.Conv3d(d1, num_classes, 1)
        self.boundary = BoundaryHead(d4)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        bmap_logits = self.boundary(x4)

        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        seg_logits = self.outc(u3)
        return seg_logits, bmap_logits
