import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_vocoder.layers import ResidualDownBlock, ResidualUpBlock, BottleneckBlock

class UNetVocoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Extract config parameters
        vocoder_config = config['vocoder']
        self.mel_bins = config['model']['mel_bins']
        self.hop_length = config['audio']['hop_length']
        self.use_f0 = vocoder_config.get('use_f0_conditioning', False)
        
        # Determine encoder and decoder channel configurations
        encoder_channels = vocoder_config.get('encoder_channels', [32, 64, 96, 128])
        decoder_channels = vocoder_config.get('decoder_channels', [96, 64, 32, 16])
        kernel_size = vocoder_config.get('kernel_size', 5)
        
        # Initial input projection (mel + noise + optional f0)
        input_channels = self.mel_bins + 1  # Mel + Noise
        if self.use_f0:
            input_channels += 1  # Add channel for F0
            
        self.input_proj = nn.Conv1d(input_channels, encoder_channels[0],
                                    kernel_size=kernel_size, padding=kernel_size//2)
        
        # Encoder blocks (downsampling path) with residual connections
        self.down_blocks = nn.ModuleList()
        for i in range(len(encoder_channels)-1):
            self.down_blocks.append(
                ResidualDownBlock(encoder_channels[i], encoder_channels[i+1], kernel_size)
            )
        
        # Enhanced bottleneck with self-attention
        self.bottleneck = BottleneckBlock(encoder_channels[-1], encoder_channels[-1], kernel_size)
        
        # Decoder blocks (upsampling path) with residual connections
        self.up_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            self.up_blocks.append(
                ResidualUpBlock(encoder_channels[-i-1], decoder_channels[i], kernel_size)
            )
        
        # Non-adjacent skip connections (connecting encoder to decoder at different levels)
        # Implemented as 1x1 convs to adapt feature dimensions
        self.non_adjacent_skips = nn.ModuleList()
        if len(encoder_channels) >= 3 and len(decoder_channels) >= 3:
            # Add connections between non-adjacent levels
            # e.g., skip from 1st encoder to 3rd decoder
            num_skip_connections = min(len(encoder_channels) - 2, len(decoder_channels) - 2)
            for i in range(num_skip_connections):
                self.non_adjacent_skips.append(
                    nn.Conv1d(encoder_channels[i], decoder_channels[i+2], kernel_size=1)
                )
        
        # Output convolution before upsampling
        self.pre_output_conv = nn.Sequential(
            nn.Conv1d(decoder_channels[-1], decoder_channels[-1], kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(decoder_channels[-1], decoder_channels[-1], kernel_size=1)
        )
        
        # Final audio upsampling layer
        self.audio_upsampler = nn.ConvTranspose1d(
            decoder_channels[-1], 
            1, 
            kernel_size=self.hop_length * 2,
            stride=self.hop_length,
            padding=self.hop_length // 2
        )
        
        # Final activation
        self.output_activation = nn.Tanh()
    
    def forward(self, mel, noise, f0=None):
        """
        Forward pass through the Enhanced U-Net vocoder
        
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
                raise ValueError("F0 conditioning is enabled, but f0 tensor was not provided to forward method.")
            # Ensure f0 has shape [B, T, 1] before transpose
            if f0.dim() != 3 or f0.shape[0] != B or f0.shape[1] != T or f0.shape[2] != 1:
                # Attempt to reshape if possible (e.g., from [B, T])
                if f0.dim() == 2 and f0.shape[0] == B and f0.shape[1] == T:
                    print(f"Warning: Reshaping f0 from {f0.shape} to {[B, T, 1]}")
                    f0 = f0.unsqueeze(-1)
                else:
                    raise ValueError(f"Expected f0 shape {[B, T, 1]}, but got {f0.shape}")
            f0 = f0.transpose(1, 2)  # [B, 1, T]
            inputs_to_cat.append(f0)
            
        x = torch.cat(inputs_to_cat, dim=1)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Encoder path with skip connections
        skip_connections = []
        non_adjacent_features = []
        
        for i, block in enumerate(self.down_blocks):
            x, skip = block(x)
            skip_connections.append(skip)
            
            # Store features for non-adjacent skip connections
            if hasattr(self, 'non_adjacent_skips') and i < len(self.non_adjacent_skips):
                non_adjacent_features.append(skip)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        
        # Decoder path using skip connections
        for i, (block, skip) in enumerate(zip(self.up_blocks, reversed(skip_connections))):
            x = block(x, skip)
            
            # Add non-adjacent skip connection features if available
            if hasattr(self, 'non_adjacent_skips') and i >= 2 and i-2 < len(non_adjacent_features):
                # Project features from encoder to match decoder dimensions
                na_skip = self.non_adjacent_skips[i-2](non_adjacent_features[i-2])
                
                # Ensure dimensions match for addition
                if na_skip.size(2) != x.size(2):
                    na_skip = F.interpolate(na_skip, size=x.size(2), mode='linear', align_corners=False)
                
                # Add the non-adjacent features
                x = x + na_skip
        
        # Pre-output convolution
        x = self.pre_output_conv(x)
        
        # Audio upsampling from mel frame rate to audio sample rate
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