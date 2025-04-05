import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleLayer(nn.Module):
    """
    Upsampling layer using transposed convolution
    """
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=scale_factor * 2,
            stride=scale_factor,
            padding=scale_factor // 2
        )
        self.norm = nn.InstanceNorm1d(channels)
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, channels, time]
        Returns:
            Upsampled features [batch, channels, time*scale_factor]
        """
        x = self.transposed_conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class MelDecoderModule(nn.Module):
    """
    Decoder module that upsamples features and projects to mel spectrogram
    """
    def __init__(self, input_dim, mel_bins=80, upsampling_factors=[2, 2]):
        super().__init__()
        self.input_dim = input_dim
        self.mel_bins = mel_bins
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for scale in upsampling_factors:
            self.upsample_layers.append(UpsampleLayer(input_dim, scale))
        
        # Total upsampling factor
        self.total_upsampling = 1
        for factor in upsampling_factors:
            self.total_upsampling *= factor
        
        # Smoothing convolution
        self.smooth_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=3,
            padding=1
        )
        
        # Final projection to mel bins
        self.final_proj = nn.Conv1d(
            in_channels=input_dim,
            out_channels=mel_bins,
            kernel_size=1
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, time, channels]
        Returns:
            Mel spectrogram [batch, mel_bins, time*total_upsampling]
        """
        # [batch, time, channels] -> [batch, channels, time]
        x = x.transpose(1, 2)
        
        # Apply upsampling
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Apply smoothing
        x = self.smooth_conv(x)
        x = F.relu(x)
        
        # Project to mel bins
        mel = self.final_proj(x)
        
        # Apply tanh activation to constrain output range
        mel = torch.tanh(mel)
        
        return mel