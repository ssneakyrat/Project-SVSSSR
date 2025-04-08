import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DownBlock, UpBlock, BottleneckBlock

class UNetVocoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Extract config parameters
        vocoder_config = config['vocoder']
        self.mel_bins = config['model']['mel_bins']
        
        # Determine encoder and decoder channel configurations
        encoder_channels = vocoder_config.get('encoder_channels', [32, 64, 96, 128])
        decoder_channels = vocoder_config.get('decoder_channels', [96, 64, 32, 16])
        kernel_size = vocoder_config.get('kernel_size', 5)
        
        # Initial input projection (mel + noise)
        self.input_proj = nn.Conv1d(self.mel_bins + 1, encoder_channels[0], 
                                   kernel_size=kernel_size, padding=kernel_size//2)
        
        # Encoder blocks (downsampling path)
        self.down_blocks = nn.ModuleList()
        for i in range(len(encoder_channels)-1):
            self.down_blocks.append(
                DownBlock(encoder_channels[i], encoder_channels[i+1], kernel_size)
            )
        
        # Bottleneck
        self.bottleneck = BottleneckBlock(encoder_channels[-1], encoder_channels[-1], kernel_size)
        
        # Decoder blocks (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            self.up_blocks.append(
                UpBlock(encoder_channels[-i-1], decoder_channels[i], kernel_size)
            )
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv1d(decoder_channels[-1], 1, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, mel, noise):
        """
        Forward pass through the U-Net vocoder
        
        Args:
            mel: Mel spectrogram [B, T, M]
            noise: Random noise [B, T, 1]
            
        Returns:
            waveform: Generated audio waveform [B, T, 1]
        """
        # Process inputs
        # Reshape mel: [B, T, M] → [B, M, T]
        mel = mel.transpose(1, 2)
        # Reshape noise: [B, T, 1] → [B, 1, T]
        noise = noise.transpose(1, 2)
        
        # Concatenate along channel dimension
        x = torch.cat([mel, noise], dim=1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Encoder path with skip connections
        skip_connections = []
        for block in self.down_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path using skip connections
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)
        
        # Output layer
        waveform = self.output_conv(x)
        
        # Reshape output: [B, 1, T] → [B, T, 1]
        waveform = waveform.transpose(1, 2)
        
        return waveform