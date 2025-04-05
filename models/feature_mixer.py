import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution for efficient feature mixing
    """
    def __init__(self, channels, kernel_size, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        # Depthwise convolution (one filter per input channel)
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels
        )
        
        # Pointwise convolution to mix channels (1x1 convolution)
        self.pointwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, channels, time]
        Returns:
            Mixed features [batch, channels, time]
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class FeatureMixingModule(nn.Module):
    """
    Efficient feature mixing using depth-wise separable convolutions
    """
    def __init__(self, channels, kernel_size=3, use_depth_wise=True):
        super().__init__()
        self.use_depth_wise = use_depth_wise
        
        if use_depth_wise:
            self.conv1 = DepthwiseSeparableConv1d(channels, kernel_size)
            self.conv2 = DepthwiseSeparableConv1d(channels, kernel_size)
        else:
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Channel-wise MLP (implemented as 1x1 convolutions)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, time, channels]
        Returns:
            Mixed features [batch, time, channels]
        """
        # [batch, time, channels] -> [batch, channels, time]
        x_t = x.transpose(1, 2)
        
        # First mixing block
        residual = x_t
        x_conv = self.conv1(x_t)
        
        # [batch, channels, time] -> [batch, time, channels] for layer norm
        x_norm = x_conv.transpose(1, 2)
        x_norm = self.norm1(x_norm)
        
        # Back to [batch, channels, time]
        x_norm = x_norm.transpose(1, 2)
        x_norm = F.gelu(x_norm)
        
        # Residual connection
        x_t = residual + x_norm
        
        # Channel-wise MLP with residual
        residual = x_t
        x_mlp = self.channel_mlp(x_t)
        x_t = residual + x_mlp
        
        # Second mixing block
        residual = x_t
        x_conv = self.conv2(x_t)
        
        # [batch, channels, time] -> [batch, time, channels] for layer norm
        x_norm = x_conv.transpose(1, 2)
        x_norm = self.norm2(x_norm)
        
        # Back to [batch, channels, time]
        x_norm = x_norm.transpose(1, 2)
        x_norm = F.gelu(x_norm)
        
        # Residual connection
        x_t = residual + x_norm
        
        # [batch, channels, time] -> [batch, time, channels]
        output = x_t.transpose(1, 2)
        
        return output