import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ResidualConvBlock2D(nn.Module):
    """Residual Convolutional Block with 2D convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        
        # Create residual connection if channel dimensions differ
        self.residual = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Add residual connection
        x = x + self.residual(identity)
        
        x = self.activation2(x)
        return x

class TransposedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class LowResModel(nn.Module):
    """Low-Resolution Model (20×216)
    Processes input features through 1D convolutions to create low-resolution mel spectrogram.
    """
    def __init__(self, input_dim, channels_list, output_dim, output_time_frames):
        super().__init__()
        self.output_time_frames = output_time_frames
        
        # Initial 1x1 conv to match input channels
        self.reshape = nn.Conv1d(input_dim, channels_list[0], kernel_size=1)
        
        # Create sequential conv blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.blocks.append(ConvBlock1D(
                channels_list[i], 
                channels_list[i+1]
            ))
        
        # Final projection to output dim (mel bins)
        self.output_proj = nn.Conv1d(channels_list[-1], output_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.reshape(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x) # Shape [B, output_dim, T_original]

        # Resize time dimension
        if x.shape[-1] != self.output_time_frames:
            x = F.interpolate(x, size=self.output_time_frames, mode='linear', align_corners=False)
            
        return x

class MidResUpsampler(nn.Module):
    """Mid-Resolution Upsampler (40×432)
    Upsamples low-resolution output to mid-resolution with improved architecture.
    - Adds skip connections from Stage 1
    - Uses separate operations for time and frequency upsampling
    - Deeper network with more layers
    - Improved weight initialization
    """
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Separate upsampling for time and frequency dimensions
        # First handle time dimension upsampling
        self.time_upsampler = TransposedConvBlock2D(
            1,                      # Input channels (from single-channel 3D tensor)
            channels_list[0] // 2,  # Intermediate channels
            kernel_size=(3, 4),     # Kernel size (freq, time)
            stride=(1, 2),          # Only double time dimension
            padding=(1, 1)
        )
        
        # Then handle frequency dimension upsampling
        self.freq_conv = ConvBlock2D(channels_list[0] // 2, channels_list[0] // 2, kernel_size=3, padding=1)
        
        # Feature fusion after upsampling
        self.fusion = ConvBlock2D(
            channels_list[0] // 2, 
            channels_list[0],
            kernel_size=3, 
            padding=1
        )
        
        # Direct skip connection from Stage 1 through a projection layer
        self.skip_projection = nn.Conv2d(1, channels_list[0], kernel_size=1)
        
        # Deeper processing blocks with residual connections
        self.blocks = nn.ModuleList()
        self.intermediate_skip_projections = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            in_ch = channels_list[i]
            out_ch = channels_list[i+1]
            self.blocks.append(ResidualConvBlock2D(
                in_ch,
                out_ch,
                kernel_size=5 if i == 0 else 3,  # Larger kernel for first block
                padding=2 if i == 0 else 1
            ))
            
            # Check if an intermediate skip connection will be added for this block index later
            # Condition matches the forward pass: i > 0 and i % 2 == 0 and i < len(self.blocks) - 1
            # Note: len(self.blocks) == len(channels_list) - 1
            if i > 0 and i % 2 == 0 and i < (len(channels_list) - 1):
                # Input channels for projection are from the tensor stored 2 steps ago
                proj_in_ch = channels_list[i-2]
                # Output channels for projection must match the output of the current block (block[i])
                proj_out_ch = out_ch # which is channels_list[i+1]
                
                if proj_in_ch != proj_out_ch:
                    self.intermediate_skip_projections.append(
                        nn.Conv2d(proj_in_ch, proj_out_ch, kernel_size=1)
                    )
                else:
                    # Append Identity if channels already match
                    self.intermediate_skip_projections.append(nn.Identity())
        
        # Final refinement layer
        self.refinement = ConvBlock2D(
            channels_list[-1], 
            channels_list[-1] // 2,
            kernel_size=3,
            padding=1
        )
        
        # Final projection
        self.output_proj = nn.Conv2d(channels_list[-1] // 2, 1, kernel_size=1)
        
        # Apply proper weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better gradient flow"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Store the original input for skip connection
        input_features = x
        
        # Reshape to [B, 1, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Shape [B, 1, 20, 216]
        
        # Apply time dimension upsampling
        x = self.time_upsampler(x)  # Shape [B, channels/2, 20, 432]
        
        # Apply frequency dimension upsampling
        x = self.freq_conv(x) # Apply convolution first
        # Now apply frequency upsampling dynamically
        target_freq_dim = self.output_dim
        current_time_dim = x.shape[-1]
        x = F.interpolate(
            x,
            size=(target_freq_dim, current_time_dim),
            mode='bilinear',
            align_corners=False
        ) # Shape [B, channels/2, 40, 432]
        
        # Apply feature fusion
        x = self.fusion(x)  # Shape [B, channels, 40, 432]
        
        # Add skip connection from Stage 1
        if input_features.dim() == 3:  # [B, 20, 216]
            # Prepare skip connection - first upsample time dimension
            skip = F.interpolate(
                input_features, 
                size=x.shape[-1],  # Match time dimension of upsampled feature
                mode='linear', 
                align_corners=False
            )  # Shape [B, 20, 432]
            
            # Transpose to match channel/frequency dimensions for 2D operations
            skip = skip.transpose(1, 2)  # Shape [B, 432, 20]
            
            # Upsample frequency dimension
            skip = F.interpolate(
                skip.unsqueeze(1),
                size=(x.shape[2], x.shape[3]),  # Match both dimensions
                mode='bilinear',
                align_corners=False
            )  # Shape [B, 1, 40, 432]
            
            # Project skip connection to proper channel dimension
            skip = self.skip_projection(skip)  # Shape [B, channels, 40, 432]
            
            # Add skip to main path with scaling to prevent overwhelming
            x = x + 0.3 * skip  # Scaled addition helps with stability
        
        # Process through deeper residual blocks
        intermediates = []
        skip_proj_idx = 0 # Index for projection layers
        for i, block in enumerate(self.blocks):
            intermediates.append(x)
            x = block(x)
            
            # Add intermediate skip connections every 2 blocks for deeper networks
            if i > 0 and i % 2 == 0 and i < len(self.blocks) - 1:
                # Retrieve skip tensor from 2 steps ago
                skip_tensor = intermediates[i-2]
                # Project skip tensor to match current channels using the corresponding projection layer
                projected_skip = self.intermediate_skip_projections[skip_proj_idx](skip_tensor)
                skip_proj_idx += 1 # Increment index for the next projection layer
                # Add projected skip from earlier in the network
                x = x + 0.5 * projected_skip
        
        # Apply final refinement
        x = self.refinement(x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x

class HighResUpsampler(nn.Module):
    """High-Resolution Upsampler (80×864)
    Upsamples mid-resolution output to high-resolution (full) with enhanced capacity.
    Includes skip connections to preserve information flow.
    """
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Improved upsampler with better initialization
        self.upsampler = TransposedConvBlock2D(
            1, 
            channels_list[0],
            kernel_size=4,
            stride=2, 
            padding=1
        )
        
        # Processing blocks with more capacity and residual connections
        self.blocks = nn.ModuleList()
        
        # Store intermediate activations for skip connections
        self.intermediate_layers = nn.ModuleList()
        
        # First block after upsampling
        self.blocks.append(
            ResidualConvBlock2D(
                channels_list[0], 
                channels_list[1],
                kernel_size=5,  # Larger kernel for better context
                padding=2       # Adjust padding for kernel size
            )
        )
        
        # Add deeper network with skip connections for remaining blocks
        for i in range(1, len(channels_list) - 2):
            # Add intermediate layer for feature fusion (1x1 conv for skip connection)
            if i % 2 == 0:  # Add skip connections every 2 layers
                self.intermediate_layers.append(
                    nn.Conv2d(channels_list[i-1], channels_list[i+1], kernel_size=1)
                )
            
            # Add residual block
            self.blocks.append(
                ResidualConvBlock2D(
                    channels_list[i], 
                    channels_list[i+1],
                    kernel_size=3 if i > 2 else 5,  # Larger kernels for early layers
                    padding=1 if i > 2 else 2       # Matching padding
                )
            )
        
        # Final refinement layer
        self.refinement = ConvBlock2D(
            channels_list[-2], 
            channels_list[-2],
            kernel_size=3,
            padding=1
        )
        
        # Final projection to single channel
        self.output_proj = nn.Conv2d(channels_list[-2], 1, kernel_size=1)
        
    def forward(self, x):
        # Input is [B, 1, C, T] from Stage 2
        
        # Apply upsampling
        x = self.upsampler(x)
        
        # Process through blocks with skip connections
        skip_idx = 0
        features = []
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Store features for skip connections
            if i < len(self.blocks) - 1 and i % 2 == 0:
                features.append(x)
            
            # Apply skip connection
            if i > 0 and i % 2 == 0 and skip_idx < len(self.intermediate_layers):
                skip_feature = self.intermediate_layers[skip_idx](features[skip_idx])
                x = x + skip_feature  # Add skip connection
                skip_idx += 1
        
        # Apply final refinement
        x = self.refinement(x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x