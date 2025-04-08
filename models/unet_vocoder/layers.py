import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(2)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        skip = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(skip)))
        return self.pool(x), skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv1 = nn.Conv1d(in_channels*2, in_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Handle case where dimensions don't match exactly due to odd lengths
        if x.size(2) != skip.size(2):
            x = F.pad(x, (0, skip.size(2) - x.size(2)))
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x