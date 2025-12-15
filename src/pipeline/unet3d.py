import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """
    GroupNorm helper that ensures num_groups divides num_channels.
    """
    g = min(num_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class DoubleConv3D(nn.Module):
    """
    (Conv3d -> GroupNorm -> ReLU) * 2
    """
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            _gn(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """
    Downsampling block: MaxPool3d -> DoubleConv3D
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up3D(nn.Module):
    """
    Upsampling block:
    - Trilinear upsampling
    - Skip connection concatenation
    - DoubleConv3D
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x, skip):
        # Upsample to match skip connection spatial size
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """
    Baseline 3D U-Net for Vbp -> CT_refined.

    Expects input:  (B, 1, D, H, W) = (B, 1, 128, 160, 160)
    Outputs:        (B, 1, D, H, W) = (B, 1, 128, 160, 160)
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()

        c = base_channels

        # Encoder
        self.inc = DoubleConv3D(in_channels, c)     # (B, 16, 128,160,160)
        self.down1 = Down3D(c, c * 2)               # (B, 32, 64, 80, 80)
        self.down2 = Down3D(c * 2, c * 4)           # (B, 64, 32, 40, 40)
        self.down3 = Down3D(c * 4, c * 8)           # (B, 128,16, 20, 20)

        # Bottleneck
        self.bot = DoubleConv3D(c * 8, c * 16)      # (B, 256,16, 20, 20)

        # Decoder
        self.up3 = Up3D(c * 16 + c * 8, c * 8)      # (256+128 -> 128)
        self.up2 = Up3D(c * 8 + c * 4, c * 4)       # (128+64  -> 64)
        self.up1 = Up3D(c * 4 + c * 2, c * 2)       # (64+32   -> 32)
        self.up0 = Up3D(c * 2 + c, c)               # (32+16   -> 16)

        # Output head
        self.outc = nn.Conv3d(c, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Bottleneck
        xb = self.bot(x3)

        # Decoder
        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        # Output
        x = self.outc(x)
        x = self.act(x)

        return x
