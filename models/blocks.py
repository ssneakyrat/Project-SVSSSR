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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_leaky_relu=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Use LeakyReLU for better gradient flow if specified
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class UpsampleBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2, use_leaky_relu=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBlock2D(in_channels, out_channels, kernel_size, stride, padding, use_leaky_relu)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class TransposedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_leaky_relu=False):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Use LeakyReLU for better gradient flow if specified
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_leaky_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Use LeakyReLU for better gradient flow if specified
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return self.activation(x)

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
        
        # Use LeakyReLU for all components in the high-res upsampler
        self.use_leaky_relu = True
        
        # Keep original channel sizes for checkpoint compatibility 
        self.upsample = TransposedConvBlock2D(1, channels_list[0], use_leaky_relu=self.use_leaky_relu)
        
        # Create main processing blocks
        self.blocks = nn.ModuleList()
        
        # Create the blocks with the exact same structure as original, but with LeakyReLU
        for i in range(len(channels_list) - 1):
            self.blocks.append(
                ConvBlock2D(channels_list[i], channels_list[i+1], use_leaky_relu=self.use_leaky_relu)
            )
        
        # Add a final projection layer
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Save input for residual connection
        identity = x
        
        # Input shape: [B, 1, C, T]
        x = self.upsample(x)
        
        # Process through main blocks
        for i, block in enumerate(self.blocks):
            # Save the intermediate output for a local residual connection
            if i > 0:  # Skip first block for stability
                block_input = x
                x = block(x)
                
                # Add a local residual connection with a small weight
                if x.shape == block_input.shape:
                    x = x + 0.1 * block_input
            else:
                x = block(x)
        
        # Final processing
        x = self.output_proj(x)
        
        # Add global residual connection with proper upsampling
        if identity.shape[2:] != x.shape[2:]:
            identity = F.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Add the residual connection (with a smaller weight to prevent dominating)
        x = x + 0.1 * identity
        
        # Output shape: [B, 1, C, T]
        return x