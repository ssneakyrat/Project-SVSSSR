import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ResidualConvBlock2D(nn.Module):
    """Residual Convolutional Block with 2D convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        
        # Create residual connection if channel dimensions differ
        self.residual = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Add residual connection
        x = x + self.residual(identity)
        
        x = self.activation2(x)
        return x

class TransposedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class LowResModel(nn.Module):
    """Low-Resolution Model (20×216)
    Processes input features through 1D convolutions to create low-resolution mel spectrogram.
    """
    def __init__(self, input_dim, channels_list, output_dim, output_time_frames):
        super().__init__()
        self.output_time_frames = output_time_frames
        
        # Initial 1x1 conv to match input channels
        self.reshape = nn.Conv1d(input_dim, channels_list[0], kernel_size=1)
        
        # Create sequential conv blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock1D(
                channels_list[i], 
                channels_list[i+1]
            ))
        
        # Final projection to output dim (mel bins)
        self.output_proj = nn.Conv1d(channels_list[-1], output_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.reshape(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x) # Shape [B, output_dim, T_original]

        # Resize time dimension
        if x.shape[-1] != self.output_time_frames:
            x = F.interpolate(x, size=self.output_time_frames, mode='linear', align_corners=False)
            
        return x

class MidResUpsampler(nn.Module):
    """Mid-Resolution Upsampler (40×432)
    Upsamples low-resolution output to mid-resolution.
    """
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        self.output_dim = output_dim

        # Upsampler: Use stride=(1, 2) to only upsample time
        self.upsampler = TransposedConvBlock2D(
            1,
            channels_list[0],
            kernel_size=(3, 4),  # Kernel (Freq, Time)
            stride=(1, 2),       # Stride (Freq, Time) - Only double time
            padding=(1, 1)       # Padding (Freq, Time)
        )
        
        # Processing blocks with residual connections for better gradient flow
        self.blocks = nn.ModuleList()
        current_channels = channels_list[0]
        for i in range(len(channels_list) - 1):
            out_ch = channels_list[i+1]
            self.blocks.append(ResidualConvBlock2D(
                current_channels,
                out_ch
            ))
            current_channels = out_ch

        # Final projection
        self.output_proj = nn.Conv2d(current_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Input is [B, C, T] from Stage 1, shape [B, 20, 216]

        # Reshape to [B, 1, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1) # Shape [B, 1, 20, 216]
        
        # Apply upsampling (only time)
        x = self.upsampler(x) # Shape [B, channels_list[0], 20, 432]
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Final projection
        x = self.output_proj(x) # Shape [B, 1, 20, 432]

        # Resize frequency dimension to self.output_dim (40)
        if x.shape[2] != self.output_dim:
             x = F.interpolate(x, size=(self.output_dim, x.shape[3]), mode='bilinear', align_corners=False)
             
        return x

class HighResUpsampler(nn.Module):
    """High-Resolution Upsampler (80×864)
    Upsamples mid-resolution output to high-resolution (full) with enhanced capacity.
    """
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Improved upsampler with better initialization
        self.upsampler = TransposedConvBlock2D(
            1, 
            channels_list[0],
            kernel_size=4,
            stride=2, 
            padding=1
        )
        
        # Processing blocks with more capacity and residual connections
        self.blocks = nn.ModuleList()
        
        # First half of blocks with larger kernels (5x5) for better receptive field
        mid_point = len(channels_list) // 2
        for i in range(mid_point):
            self.blocks.append(
                ResidualConvBlock2D(
                    channels_list[i], 
                    channels_list[i+1],
                    kernel_size=5,  # Larger kernel for better context
                    padding=2       # Adjust padding for kernel size
                )
            )
        
        # Second half of blocks with standard kernels (3x3)
        for i in range(mid_point, len(channels_list) - 1):
            self.blocks.append(
                ResidualConvBlock2D(
                    channels_list[i], 
                    channels_list[i+1]
                )
            )
        
        # Final refinement layer with standard kernel
        self.refinement = ConvBlock2D(
            channels_list[-1], 
            channels_list[-1],
            kernel_size=3,
            padding=1
        )
        
        # Final projection to single channel
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Input is [B, 1, C, T] from Stage 2
        
        # Apply upsampling
        x = self.upsampler(x)
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final refinement
        x = self.refinement(x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x