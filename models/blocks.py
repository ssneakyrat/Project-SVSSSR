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
        # Permute output from (B, F', T) to (B, T, F')
        return x.permute(0, 2, 1)

class MidResUpsampler(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Upsample only frequency dimension (dim 2), preserve time (dim 3)
        # Input shape expected by Upsample: [B, C, F, T]
        self.upsample = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False)
        
        # First ConvBlock adapts from 1 channel (after unsqueeze) to channels_list[0]
        self.initial_conv = ConvBlock2D(1, channels_list[0])
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock2D(channels_list[i], channels_list[i+1]))
            
        # Final projection back to 1 channel, preserving F and T dimensions
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [B, T, F_in]
        # Permute to [B, F_in, T]
        x = x.permute(0, 2, 1)
        # Unsqueeze to [B, 1, F_in, T] for 2D operations
        x = x.unsqueeze(1)
        
        # Upsample Frequency: [B, 1, F_in, T] -> [B, 1, 2*F_in, T]
        x = self.upsample(x)
        
        # Initial convolution: [B, 1, F_mid_approx, T] -> [B, C0, F_mid_approx, T]
        x = self.initial_conv(x)
        
        # Conv blocks: [B, C0, F, T] -> [B, Clast, F, T]
        for block in self.blocks:
            x = block(x)
            
        # Output projection: [B, Clast, F, T] -> [B, 1, F_mid, T]
        x = self.output_proj(x)
        
        # Squeeze channel dim: [B, 1, F_mid, T] -> [B, F_mid, T]
        x = x.squeeze(1)
        # Permute back to [B, T, F_mid]
        return x.permute(0, 2, 1)

class HighResUpsampler(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Upsample only frequency dimension (dim 2), preserve time (dim 3)
        # Input shape expected by Upsample: [B, C, F, T]
        self.upsample = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False)
        
        # First ConvBlock adapts from 1 channel (input_dim is freq dim here, but input has 1 channel after unsqueeze)
        self.initial_conv = ConvBlock2D(1, channels_list[0])
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock2D(channels_list[i], channels_list[i+1]))
            
        # Final projection back to 1 channel, preserving F and T dimensions
        # The output_dim parameter is not directly used here, final freq dim is determined by upsampling + convs
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [B, T, F_mid]
        # Permute to [B, F_mid, T]
        x = x.permute(0, 2, 1)
        # Unsqueeze to [B, 1, F_mid, T] for 2D operations
        x = x.unsqueeze(1)
        
        # Upsample Frequency: [B, 1, F_mid, T] -> [B, 1, 2*F_mid, T]
        x = self.upsample(x)
        
        # Initial convolution: [B, 1, F_high_approx, T] -> [B, C0, F_high_approx, T]
        x = self.initial_conv(x)
        
        # Conv blocks: [B, C0, F, T] -> [B, Clast, F, T]
        for block in self.blocks:
            x = block(x)
            
        # Output projection: [B, Clast, F, T] -> [B, 1, F_high, T]
        x = self.output_proj(x)
        
        # Squeeze channel dim: [B, 1, F_high, T] -> [B, F_high, T]
        x = x.squeeze(1)
        # Permute back to [B, T, F_high]
        return x.permute(0, 2, 1)