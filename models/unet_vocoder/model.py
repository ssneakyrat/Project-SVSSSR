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
        
        # Store encoder channels for reference during forward pass
        self.encoder_channels = encoder_channels
        
        # Non-adjacent skip connections (connecting encoder to decoder at different levels)
        # Implemented as 1x1 convs to adapt feature dimensions
        self.non_adjacent_skips = nn.ModuleList()
        if len(encoder_channels) >= 3 and len(decoder_channels) >= 3:
            # Add connections between non-adjacent levels
            # e.g., skip from 1st encoder to 3rd decoder
            num_skip_connections = min(len(encoder_channels) - 2, len(decoder_channels) - 2)
            
            # We'll create these in the forward pass instead, since we need to know
            # the actual dimensions of the features
            self.use_non_adjacent_skips = True
            self.num_skip_connections = num_skip_connections
            self.decoder_channels = decoder_channels
        else:
            self.use_non_adjacent_skips = False
        
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
        
        # NEW: Autoregressive state handling
        self.use_autoregressive = vocoder_config.get('use_autoregressive', False)
        
        if self.use_autoregressive:
            # Define recurrent units for maintaining state across chunks
            # We need one for each level of the U-Net
            self.encoder_gru_layers = nn.ModuleList()
            for ch in encoder_channels:
                self.encoder_gru_layers.append(
                    nn.GRU(ch, ch, batch_first=True, bidirectional=False)
                )
            
            self.bottleneck_gru = nn.GRU(
                encoder_channels[-1], 
                encoder_channels[-1], 
                batch_first=True, 
                bidirectional=False
            )
            
            self.decoder_gru_layers = nn.ModuleList()
            for ch in decoder_channels:
                self.decoder_gru_layers.append(
                    nn.GRU(ch, ch, batch_first=True, bidirectional=False)
                )
    
    def _init_state(self, batch_size, device):
        """Initialize state for autoregressive generation"""
        encoder_states = []
        for i, layer in enumerate(self.encoder_gru_layers):
            # Hidden state shape: [num_layers * num_directions, batch_size, hidden_size]
            hidden_size = self.encoder_channels[i] if i < len(self.encoder_channels) else self.encoder_channels[-1]
            h = torch.zeros(1, batch_size, hidden_size, device=device)
            encoder_states.append(h)
        
        # Bottleneck state
        bottleneck_state = torch.zeros(1, batch_size, self.encoder_channels[-1], device=device)
        
        # Decoder states
        decoder_states = []
        for i, layer in enumerate(self.decoder_gru_layers):
            hidden_size = self.decoder_channels[i]
            h = torch.zeros(1, batch_size, hidden_size, device=device)
            decoder_states.append(h)
        
        return {
            'encoder_states': encoder_states,
            'bottleneck_state': bottleneck_state,
            'decoder_states': decoder_states
        }
    
    def forward(self, mel, noise, f0=None, state=None):
        """
        Forward pass through the Enhanced U-Net vocoder with autoregressive capabilities
        
        Args:
            mel (torch.Tensor): Mel spectrogram [B, T, M]
            noise (torch.Tensor): Random noise [B, T, 1]
            f0 (torch.Tensor, optional): Aligned F0 contour [B, T, 1]. Defaults to None.
            state (dict, optional): Autoregressive state from previous chunk. Defaults to None.
            
        Returns:
            waveform: Generated audio waveform [B, T*hop_length, 1]
            new_state (optional): Updated autoregressive state for next chunk
        """
        # Check the shapes
        B, T, M = mel.shape
        B_noise, T_noise, C_noise = noise.shape
        
        # Initialize state if not provided and autoregressive mode is enabled
        if self.use_autoregressive and state is None:
            state = self._init_state(B, mel.device)
        
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
        
        # Encoder path with skip connections and optional autoregressive state
        skip_connections = []
        non_adjacent_features = []
        new_encoder_states = []
        
        for i, block in enumerate(self.down_blocks):
            x, skip = block(x)
            skip_connections.append(skip)
            
            # Apply autoregressive update if enabled
            if self.use_autoregressive:
                # Reshape for GRU: [B, C, T] -> [B, T, C]
                x_reshaped = x.transpose(1, 2)
                
                # Get previous state
                prev_state = state['encoder_states'][i]
                
                # Apply GRU
                x_recurrent, new_state_i = self.encoder_gru_layers[i](x_reshaped, prev_state)
                
                # Reshape back: [B, T, C] -> [B, C, T]
                x = x_recurrent.transpose(1, 2)
                
                # Store new state
                new_encoder_states.append(new_state_i)
            
            # Store features for non-adjacent skip connections
            if self.use_non_adjacent_skips and i < self.num_skip_connections:
                non_adjacent_features.append(skip)
        
        # Bottleneck with attention and optional autoregressive state
        x = self.bottleneck(x)
        
        if self.use_autoregressive:
            # Apply recurrent bottleneck
            x_reshaped = x.transpose(1, 2)
            prev_bottleneck_state = state['bottleneck_state']
            x_recurrent, new_bottleneck_state = self.bottleneck_gru(x_reshaped, prev_bottleneck_state)
            x = x_recurrent.transpose(1, 2)
        
        # Create dynamic non-adjacent skip connections based on actual feature dimensions
        if self.use_non_adjacent_skips and len(self.non_adjacent_skips) == 0 and len(non_adjacent_features) > 0:
            for i in range(self.num_skip_connections):
                # Get the actual input channel dimension from the stored features
                in_channels = non_adjacent_features[i].size(1)
                out_channels = self.decoder_channels[i+2]
                
                # Create the 1x1 conv with the correct dimensions
                self.non_adjacent_skips.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1).to(x.device)
                )
        
        # Decoder path using skip connections and optional autoregressive state
        new_decoder_states = []
        
        for i, (block, skip) in enumerate(zip(self.up_blocks, reversed(skip_connections))):
            x = block(x, skip)
            
            # Apply autoregressive update if enabled
            if self.use_autoregressive:
                # Reshape for GRU: [B, C, T] -> [B, T, C]
                x_reshaped = x.transpose(1, 2)
                
                # Get previous state
                prev_state = state['decoder_states'][i]
                
                # Apply GRU
                x_recurrent, new_state_i = self.decoder_gru_layers[i](x_reshaped, prev_state)
                
                # Reshape back: [B, T, C] -> [B, C, T]
                x = x_recurrent.transpose(1, 2)
                
                # Store new state
                new_decoder_states.append(new_state_i)
            
            # Add non-adjacent skip connection features if available
            if self.use_non_adjacent_skips and i >= 2 and i-2 < len(non_adjacent_features):
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
        
        # Return waveform with state if autoregressive mode is enabled
        if self.use_autoregressive:
            new_state = {
                'encoder_states': new_encoder_states,
                'bottleneck_state': new_bottleneck_state,
                'decoder_states': new_decoder_states
            }
            return waveform, new_state
        else:
            return waveform