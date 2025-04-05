import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_leaky_relu=False, use_layer_norm=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Option to use layer norm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            # Will apply in forward pass
            self.norm = None
        else:
            self.norm = nn.BatchNorm1d(out_channels)
        
        # Option to use LeakyReLU
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_layer_norm:
            # Reshape for layer norm (N, C, L) -> (N, C, 1, L) -> apply norm -> reshape back
            N, C, L = x.shape
            x = x.unsqueeze(2)  # (N, C, 1, L)
            x = nn.functional.layer_norm(x, [1, L])
            x = x.squeeze(2)  # (N, C, L)
        else:
            x = self.norm(x)
            
        x = self.activation(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_leaky_relu=False, use_layer_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Option to use LayerNorm instead of BatchNorm to reduce artifacts
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            # LayerNorm is applied after reshaping to (N, C, H*W)
            self.norm = None  # Will apply LayerNorm in forward
        else:
            self.norm = nn.BatchNorm2d(out_channels)
        
        # Use LeakyReLU for better gradient flow if specified
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        
        if self.use_layer_norm:
            # Reshape to (N, C, H*W) for LayerNorm, then back to (N, C, H, W)
            N, C, H, W = x.shape
            x = x.reshape(N, C, -1)
            x = nn.functional.layer_norm(x, [x.size(-1)])
            x = x.reshape(N, C, H, W)
        else:
            x = self.norm(x)
            
        x = self.activation(x)
        return x

class UpsampleBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2, 
                 use_leaky_relu=False, use_layer_norm=False, mode='bilinear'):
        super().__init__()
        
        # Use bilinear upsampling instead of transposed convolution to reduce artifacts
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=False)
        self.conv = ConvBlock2D(in_channels, out_channels, kernel_size, stride, padding, 
                              use_leaky_relu=use_leaky_relu, use_layer_norm=use_layer_norm)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class TransposedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_leaky_relu=False, use_layer_norm=False):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Option to use LayerNorm instead of BatchNorm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            # LayerNorm is applied in forward pass
            self.norm = None 
        else:
            self.norm = nn.BatchNorm2d(out_channels)
        
        # Use LeakyReLU for better gradient flow if specified
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        
        if self.use_layer_norm:
            # Reshape for LayerNorm
            N, C, H, W = x.shape
            x = x.reshape(N, C, -1)
            x = nn.functional.layer_norm(x, [x.size(-1)])
            x = x.reshape(N, C, H, W)
        else:
            x = self.norm(x)
            
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_leaky_relu=False, use_layer_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Option to use LayerNorm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm1 = None
            self.norm2 = None
        else:
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        
        # Use LeakyReLU for better gradient flow if specified
        if use_leaky_relu:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        if self.use_layer_norm:
            # Apply layer norm
            N, C, H, W = x.shape
            x = x.reshape(N, C, -1)
            x = nn.functional.layer_norm(x, [x.size(-1)])
            x = x.reshape(N, C, H, W)
        else:
            x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        if self.use_layer_norm:
            # Apply layer norm
            N, C, H, W = x.shape
            x = x.reshape(N, C, -1)
            x = nn.functional.layer_norm(x, [x.size(-1)])
            x = x.reshape(N, C, H, W)
        else:
            x = self.norm2(x)
            
        # Increase residual connection influence (0.7 -> 0.8)
        x = x + 0.8 * residual
        return self.activation(x)

class LowResModel(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        self.reshape = nn.Conv1d(input_dim, channels_list[0], kernel_size=1)
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            # Use improved version with options
            self.blocks.append(ConvBlock1D(
                channels_list[i], 
                channels_list[i+1],
                use_leaky_relu=True,  # Use LeakyReLU for better gradients
                use_layer_norm=False  # Keep BatchNorm for Stage 1 (less critical)
            ))
        
        # Add a smoothing conv before output
        self.smoothing = nn.Conv1d(
            channels_list[-1], 
            channels_list[-1],
            kernel_size=5,  # Larger kernel for better smoothing
            padding=2,
            groups=channels_list[-1]  # Depthwise for efficiency
        )
        
        self.output_proj = nn.Conv1d(channels_list[-1], output_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.reshape(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Apply smoothing to reduce any frequency artifacts
        x = self.smoothing(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.output_proj(x)
        return x

class MidResUpsampler(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Use bilinear upsampling instead of transposed convolution
        self.upsample = UpsampleBlock2D(
            1, channels_list[0], 
            use_leaky_relu=True,
            use_layer_norm=True,  # Use layer norm to reduce artifacts
            scale_factor=2,
            mode='bilinear'
        )
        
        self.blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            # Use improved ConvBlock2D with options for better normalization
            self.blocks.append(ConvBlock2D(
                channels_list[i], 
                channels_list[i+1],
                use_leaky_relu=True,
                use_layer_norm=True
            ))
        
        # Add a smoothing layer before final projection
        self.smoothing = nn.Conv2d(
            channels_list[-1], 
            channels_list[-1],
            kernel_size=3, 
            padding=1,
            groups=channels_list[-1]  # Depthwise convolution for smoothing
        )
        
        # Final projection with smaller initialization
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        # Initialize with smaller weights
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        
    def forward(self, x):
        # Reshape from [B, C, T] to [B, 1, C, T]
        x = x.unsqueeze(1)
        
        # Save for potential residual connection
        identity = x
        
        x = self.upsample(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Apply smoothing
        x = self.smoothing(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.output_proj(x)
        
        # Optional light residual connection if dimensions match
        if identity.shape[2:] != x.shape[2:]:
            identity = F.interpolate(
                identity, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            # Increase residual influence from 0.05 to 0.1
            x = x + 0.1 * identity
        
        # Output shape: [B, 1, C, T]
        return x

class HighResUpsampler(nn.Module):
    def __init__(self, input_dim, channels_list, output_dim):
        super().__init__()
        
        # Use LeakyReLU for all components in the high-res upsampler
        self.use_leaky_relu = True
        
        # Use Layer Normalization to reduce artifacts
        self.use_layer_norm = True
        
        # Replace transposed convolution with bilinear upsampling + convolution
        # This will significantly reduce the checkerboard artifacts
        self.upsample = UpsampleBlock2D(
            1, channels_list[0], 
            use_leaky_relu=self.use_leaky_relu,
            use_layer_norm=self.use_layer_norm,
            scale_factor=2,
            mode='bilinear'  # Using bilinear instead of nearest for smoother results
        )
        
        # Create main processing blocks
        self.blocks = nn.ModuleList()
        
        # Create residual blocks instead of regular conv blocks
        # This will help preserve details while reducing artifacts
        for i in range(len(channels_list) - 1):
            self.blocks.append(
                ResidualBlock(
                    channels_list[i], 
                    use_leaky_relu=self.use_leaky_relu,
                    use_layer_norm=self.use_layer_norm
                )
            )
            # Add a transitional conv after each residual block to change channel dimensions
            self.blocks.append(
                ConvBlock2D(
                    channels_list[i], channels_list[i+1],
                    use_leaky_relu=self.use_leaky_relu,
                    use_layer_norm=self.use_layer_norm
                )
            )
        
        # Add smoothing conv before final projection to reduce high-frequency artifacts
        self.smoothing_conv = nn.Conv2d(
            channels_list[-1], channels_list[-1], 
            kernel_size=3, padding=1, groups=channels_list[-1]  # Depthwise conv for smoothing
        )
        
        # Add a final projection layer with INCREASED weight initialization (key fix)
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        # Initialize with LARGER weights to make output more visible (0.1 -> 0.5)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        
        # Add final activation to constrain output to [0, 1] range
        self.output_normalization = nn.Sigmoid() # Changed from Identity to Sigmoid
        
    def forward(self, x):
        # Save input for residual connection
        identity = x
        
        # Input shape: [B, 1, C, T]
        x = self.upsample(x)
        
        # Process through main blocks
        for i, block in enumerate(self.blocks):
            # Process through the block
            block_input = x
            x = block(x)
            
            # Add a local residual connection with INCREASED weight (0.05 -> 0.15)
            # Only for even indices (the ResidualBlocks)
            if i % 2 == 0 and x.shape == block_input.shape:
                x = x + 0.15 * block_input
        
        # Apply smoothing to reduce high-frequency artifacts
        x = self.smoothing_conv(x)
        x = F.leaky_relu(x, 0.2)
        
        # Final processing
        x = self.output_proj(x)
        
        # Add global residual connection with proper upsampling and INCREASED weight (0.05 -> 0.3)
        if identity.shape[2:] != x.shape[2:]:
            # Use bilinear interpolation for smoother upsampling
            identity = F.interpolate(
                identity, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Add the residual connection with a LARGER weight (0.05 -> 0.3)
        x = x + 0.3 * identity
        
        # Apply final activation (Sigmoid)
        x = self.output_normalization(x) # Applies Sigmoid now
        
        # Output shape: [B, 1, C, T]
        return x