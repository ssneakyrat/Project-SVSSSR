import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpsampleBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBlock2D(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class TransposedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)

class LowResModel(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        self.reshape = nn.Conv1d(input_dim, channels_list[0], kernel_size=1)
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock1D(channels_list[i], channels_list[i+1]))
        
        self.output_proj = nn.Conv1d(channels_list[-1], output_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.reshape(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        return x

class MidResUpsampler(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        self.upsample = TransposedConvBlock2D(1, channels_list[0])
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock2D(channels_list[i], channels_list[i+1]))
        
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Reshape from [B, C, T] to [B, 1, C, T]
        x = x.unsqueeze(1)
        
        x = self.upsample(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        
        # Output shape: [B, 1, C, T]
        return x

class HighResUpsampler(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        self.upsample = TransposedConvBlock2D(1, channels_list[0])
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock2D(channels_list[i], channels_list[i+1]))
        
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [B, 1, C, T]
        x = self.upsample(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        
        # Output shape: [B, 1, C, T]
        return x