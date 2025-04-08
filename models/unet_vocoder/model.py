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
        self.hop_length = config['audio']['hop_length']
        
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
        
        # Output convolution before upsampling
        self.pre_output_conv = nn.Conv1d(decoder_channels[-1], decoder_channels[-1], kernel_size=1)
        
        # Final audio upsampling layer - NEW!
        # This is the key change: explicit upsampling from mel frame rate to audio sample rate
        self.audio_upsampler = nn.ConvTranspose1d(
            decoder_channels[-1], 
            1, 
            kernel_size=self.hop_length * 2,  # Wider kernel for smoother upsampling
            stride=self.hop_length,
            padding=self.hop_length // 2
        )
        
        # Final activation
        self.output_activation = nn.Tanh()
    
    def forward(self, mel, noise):
        """
        Forward pass through the U-Net vocoder
        
        Args:
            mel: Mel spectrogram [B, T, M]
            noise: Random noise [B, T, 1]
            
        Returns:
            waveform: Generated audio waveform [B, T*hop_length, 1]
        """
        # Check the shapes
        B, T, M = mel.shape
        B_noise, T_noise, C_noise = noise.shape
        
        # Ensure time dimensions match
        if T_noise != T:
            if T_noise % T == 0:
                # If noise time dimension is a multiple of mel time dimension
                factor = T_noise // T
                # Downsample by averaging over windows
                noise = noise.reshape(B_noise, T, factor, C_noise).mean(dim=2)
            else:
                # Otherwise, use interpolation
                noise = noise.transpose(1, 2)  # [B, 1, T_noise]
                noise = F.interpolate(noise, size=T, mode='linear', align_corners=False)
                noise = noise.transpose(1, 2)  # [B, T, 1]
                
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
        
        # Pre-output convolution
        x = self.pre_output_conv(x)
        
        # NEW: Audio upsampling from mel frame rate to audio sample rate
        # Input: [B, C, T] at mel frame rate
        # Output: [B, 1, T*hop_length] at audio sample rate
        waveform = self.audio_upsampler(x)
        
        # Apply final activation
        waveform = self.output_activation(waveform)
        
        # Reshape output: [B, 1, T*hop_length] → [B, T*hop_length, 1]
        waveform = waveform.transpose(1, 2)
        
        # Verify dimensions
        expected_length = T * self.hop_length
        actual_length = waveform.size(1)
        
        # If there's still a small mismatch, adjust the waveform length
        if actual_length != expected_length:
            if actual_length < expected_length:
                # Pad if too short
                padding = expected_length - actual_length
                waveform = F.pad(waveform, (0, 0, 0, padding))
            else:
                # Trim if too long
                waveform = waveform[:, :expected_length, :]
        
        return waveform