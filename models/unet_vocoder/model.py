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
        self.use_f0 = vocoder_config.get('use_f0_conditioning', False) # Check if F0 conditioning is enabled
        
        # Determine encoder and decoder channel configurations
        encoder_channels = vocoder_config.get('encoder_channels', [32, 64, 96, 128])
        decoder_channels = vocoder_config.get('decoder_channels', [96, 64, 32, 16])
        kernel_size = vocoder_config.get('kernel_size', 5)
        
        # Initial input projection (mel + noise + optional f0)
        input_channels = self.mel_bins + 1 # Mel + Noise
        if self.use_f0:
            input_channels += 1 # Add channel for F0
            
        self.input_proj = nn.Conv1d(input_channels, encoder_channels[0],
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
    
    def forward(self, mel, noise, f0=None):
        """
        Forward pass through the U-Net vocoder
        
        Args:
            mel (torch.Tensor): Mel spectrogram [B, T, M]
            noise (torch.Tensor): Random noise [B, T, 1]
            f0 (torch.Tensor, optional): Aligned F0 contour [B, T, 1]. Defaults to None.
            
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
        inputs_to_cat = [mel, noise]
        if self.use_f0:
            if f0 is None:
                raise ValueError("F0 conditioning is enabled ('use_f0_conditioning: true' in config), but f0 tensor was not provided to forward method.")
            # Ensure f0 has shape [B, T, 1] before transpose
            if f0.dim() != 3 or f0.shape[0] != B or f0.shape[1] != T or f0.shape[2] != 1:
                 # Attempt to reshape if possible (e.g., from [B, T])
                 if f0.dim() == 2 and f0.shape[0] == B and f0.shape[1] == T:
                     print(f"Warning: Reshaping f0 from {f0.shape} to {[B, T, 1]}")
                     f0 = f0.unsqueeze(-1)
                 else:
                    raise ValueError(f"Expected f0 shape {[B, T, 1]}, but got {f0.shape}")
            f0 = f0.transpose(1, 2) # [B, 1, T]
            inputs_to_cat.append(f0)
            
        x = torch.cat(inputs_to_cat, dim=1)
        
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