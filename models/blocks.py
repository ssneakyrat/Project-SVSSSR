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

# Residual Block with Modulation for 1D data
class ResidualBlock1D(nn.Module):
    # Added stride parameter
    def __init__(self, in_channels, out_channels, unvoiced_embed_dim, kernel_size=3, stride=1):
        super().__init__()
        # Adjust padding based on stride to try and maintain dimensions where possible,
        # but exact length matching with stride > 1 can be tricky.
        # For stride=1, padding = kernel_size // 2 maintains length.
        # For stride=2, output_len = floor((input_len + 2*padding - kernel_size)/stride) + 1
        # Let's keep padding = kernel_size // 2 for simplicity for now.
        padding = kernel_size // 2

        # Use the passed stride in the first convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding) # Stride is 1 here
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Modulation layers: Project unvoiced embedding to gamma and beta for bn2
        # Use Conv1d to potentially capture temporal variations in modulation
        self.mod_conv = nn.Conv1d(unvoiced_embed_dim, out_channels * 2, kernel_size=1) # Output gamma and beta

        # Shortcut connection: Apply stride if stride > 1 OR if channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Use the same stride in the shortcut's convolution
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x, unvoiced_embedding):
        residual = self.shortcut(x)

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (before modulation)
        out = self.conv2(out)
        out = self.bn2(out) # Apply BN before modulation

        # Calculate modulation parameters (gamma, beta) using Conv1d approach
        # unvoiced_embedding shape: (B, U_dim, T)
        mod_params = self.mod_conv(unvoiced_embedding) # Shape: (B, 2 * out_channels, T)
        gamma, beta = torch.chunk(mod_params, 2, dim=1) # Each: (B, out_channels, T)

        # Apply modulation (FiLM-like)
        out = gamma * out + beta

        # Add residual and apply final ReLU
        out += residual
        out = self.relu(out)
        return out

# Modified LowResModel using ResidualBlock1D with modulation
class LowResModel(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim, unvoiced_embed_dim): # Added unvoiced_embed_dim
        super().__init__()
        
        self.reshape = nn.Conv1d(input_dim, channels_list[0], kernel_size=1)
        
        self.blocks = nn.ModuleList()
        current_channels = channels_list[0]
        # Iterate through all target channels to define blocks
        for i in range(len(channels_list)):
            out_channels = channels_list[i] # Target channel for this block
            # Use stride=2 for the first block to downsample time, stride=1 otherwise
            stride = 2 if i == 0 else 1
            self.blocks.append(ResidualBlock1D(current_channels, out_channels, unvoiced_embed_dim, stride=stride))
            current_channels = out_channels # Update current channels for the next block
        
        # Output projection uses the last channel size from the list
        self.output_proj = nn.Conv1d(channels_list[-1], output_dim, kernel_size=1)
        
    def forward(self, x, unvoiced_embedding): # Added unvoiced_embedding
        x = self.reshape(x)
        
        # Downsample unvoiced_embedding if the first block uses stride > 1
        cond_embedding = unvoiced_embedding
        if len(self.blocks) > 0:
            first_block_stride = self.blocks[0].conv1.stride[0] # Get stride value
            if first_block_stride > 1:
                # Use functional avg_pool1d for downsampling
                cond_embedding = F.avg_pool1d(unvoiced_embedding, kernel_size=first_block_stride, stride=first_block_stride)

        # Pass potentially downsampled embedding to each block
        for block in self.blocks:
            x = block(x, cond_embedding)
        
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