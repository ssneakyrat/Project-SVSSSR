import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(norm_type, num_channels):
    """Returns the appropriate normalization layer based on the norm_type"""
    if norm_type == "instance":
        return nn.InstanceNorm1d(num_channels)
    elif norm_type == "layer":
        return nn.GroupNorm(1, num_channels)  # LayerNorm is GroupNorm with groups=1
    else:  # Default to BatchNorm
        return nn.BatchNorm1d(num_channels)

class ResidualDownBlock(nn.Module):
    """Enhanced DownBlock with residual connections and configurable normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=5, norm_type="instance"):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = get_norm_layer(norm_type, out_channels)
        self.act1 = nn.LeakyReLU(0.2)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = get_norm_layer(norm_type, out_channels)
        self.act2 = nn.LeakyReLU(0.2)
        
        # Residual connection (1x1 conv if channel dimensions don't match)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Pooling layer for downsampling
        self.pool = nn.AvgPool1d(2)
        
    def forward(self, x):
        # Residual connection
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual and apply activation
        skip = self.act2(h + residual)
        
        # Downsample and return both downsampled and skip connection
        return self.pool(skip), skip

class ResidualUpBlock(nn.Module):
    """Enhanced UpBlock with residual connections and configurable normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=5, norm_type="instance"):
        super().__init__()
        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # Concatenated input will have 2*in_channels
        self.conv1 = nn.Conv1d(in_channels*2, in_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = get_norm_layer(norm_type, in_channels)
        self.act1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = get_norm_layer(norm_type, out_channels)
        self.act2 = nn.LeakyReLU(0.2)
        
        # Residual connection (1x1 conv to match output dimensions)
        self.residual_conv = nn.Conv1d(in_channels*2, out_channels, 1)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle case where dimensions don't match due to odd lengths
        if x.size(2) != skip.size(2):
            x = F.pad(x, (0, skip.size(2) - x.size(2)))
            
        # Concatenate with skip connection
        concat = torch.cat([x, skip], dim=1)
        
        # Save for residual connection
        residual = self.residual_conv(concat)
        
        # First conv block
        h = self.conv1(concat)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual and apply activation
        out = self.act2(h + residual)
        
        return out

class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // reduction, 1)
        self.key = nn.Conv1d(channels, channels // reduction, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        # Scale factor for dot product attention
        self.scale = (channels // reduction) ** -0.5
        
        # Output projection
        self.output_proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        # Input shape: [B, C, T]
        batch_size, channels, length = x.size()
        
        # Compute query, key, value
        q = self.query(x)  # [B, C/r, T]
        k = self.key(x)    # [B, C/r, T]
        v = self.value(x)  # [B, C, T]
        
        # Compute attention scores
        # Reshape for matrix multiplication: [B, C/r, T] -> [B, C/r, T]
        q = q.permute(0, 2, 1)  # [B, T, C/r]
        
        # Matrix multiplication: [B, T, C/r] @ [B, C/r, T] -> [B, T, T]
        attn = torch.bmm(q, k) * self.scale
        
        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights to value
        # [B, T, T] @ [B, C, T]→[B, T, C]
        out = torch.bmm(attn, v.permute(0, 2, 1))
        out = out.permute(0, 2, 1)  # [B, C, T]
        
        # Output projection
        out = self.output_proj(out)
        
        # Residual connection
        out = out + x
        
        return out

class BottleneckBlock(nn.Module):
    """Enhanced bottleneck with self-attention and residual connections"""
    def __init__(self, channels, out_channels, kernel_size=5, norm_type="instance", use_attention=True, attention_reduction=8):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = get_norm_layer(norm_type, channels)
        self.act1 = nn.LeakyReLU(0.2)
        
        # Self-attention block (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(channels, reduction=attention_reduction)
        
        # Second conv block
        self.conv2 = nn.Conv1d(channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = get_norm_layer(norm_type, out_channels)
        self.act2 = nn.LeakyReLU(0.2)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Apply self-attention if enabled
        if self.use_attention:
            h = self.attention(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual and apply activation
        out = self.act2(h + residual)
        
        return out