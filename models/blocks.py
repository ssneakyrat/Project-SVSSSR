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
    def __init__(self, input_dim, channels_list, output_dim, output_time_frames): # Added output_time_frames
        super().__init__()
        self.output_time_frames = output_time_frames # Store it
        
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
            x = F.interpolate(x, size=self.output_time_frames, mode='linear', align_corners=False) # Shape [B, output_dim, output_time_frames]
            
        return x

class MidResUpsampler(nn.Module):
    """Mid-Resolution Upsampler (40×432)
    Upsamples low-resolution output to mid-resolution.
    """
    def __init__(self, input_dim, channels_list, output_dim): # input_dim=20, output_dim=40
        super().__init__()
        self.output_dim = output_dim # Store target freq bins

        # Upsampler: Use stride=(1, 2) to only upsample time. kernel/padding adjusted.
        # Input to upsampler will be [B, 1, input_dim, T_in] -> [B, 1, 20, 216]
        # Output: [B, channels_list[0], 20, 432]
        self.upsampler = TransposedConvBlock2D(
            1,
            channels_list[0],
            kernel_size=(3, 4),  # Kernel (Freq, Time)
            stride=(1, 2),       # Stride (Freq, Time) - Only double time
            padding=(1, 1)       # Padding (Freq, Time)
        )
        
        # Processing blocks (input channels need adjustment)
        self.blocks = nn.ModuleList()
        # First block input is channels_list[0] (e.g., 32)
        current_channels = channels_list[0]
        for i in range(len(channels_list) -1): # channels_list = [32, 24, 16]
            out_ch = channels_list[i+1]
            self.blocks.append(ConvBlock2D(
                current_channels,
                out_ch
            ))
            current_channels = out_ch # Blocks: 32->24, 24->16

        # Final projection (input channels is channels_list[-1], e.g., 16)
        self.output_proj = nn.Conv2d(current_channels, 1, kernel_size=1) # Output: [B, 1, 20, 432]
    def forward(self, x):
        # Input is [B, C, T] from Stage 1, shape [B, 20, 216]

        # Reshape to [B, 1, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1) # Shape [B, 1, 20, 216]
        
        # Apply upsampling (only time)
        x = self.upsampler(x) # Shape [B, 32, 20, 432]
        
        # Process through blocks
        for block in self.blocks:
            x = block(x) # Shape [B, 16, 20, 432]
        
        # Final projection
        x = self.output_proj(x) # Shape [B, 1, 20, 432]

        # Resize frequency dimension to self.output_dim (40)
        if x.shape[2] != self.output_dim:
             x = F.interpolate(x, size=(self.output_dim, x.shape[3]), mode='bilinear', align_corners=False) # Shape [B, 1, 40, 432]
             
        return x

class HighResUpsampler(nn.Module):
    """High-Resolution Upsampler (80×864)
    Upsamples mid-resolution output to high-resolution (full).
    """
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Transposed conv for upsampling (2x)
        self.upsampler = TransposedConvBlock2D(1, channels_list[0])
        
        # Processing blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock2D(
                channels_list[i], 
                channels_list[i+1]
            ))
        
        # Final projection
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Input is [B, 1, C, T] from Stage 2
        
        # Apply upsampling
        x = self.upsampler(x)
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x