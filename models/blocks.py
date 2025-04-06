import logging # Import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__) # Get logger instance

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpsampleBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = ConvBlock2D(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class TransposedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)

# Residual Block with Modulation for 1D data
class ResidualBlock1D(nn.Module):
    # Added stride parameter
    def __init__(self, in_channels, out_channels, unvoiced_embed_dim, kernel_size=3, stride=1):
        super().__init__()
        # Adjust padding based on stride to try and maintain dimensions where possible,
        # but exact length matching with stride > 1 can be tricky.
        # For stride=1, padding = kernel_size // 2 maintains length.
        # For stride=2, output_len = floor((input_len + 2*padding - kernel_size)/stride) + 1
        # Let's keep padding = kernel_size // 2 for simplicity for now.
        padding = kernel_size // 2

        # Use the passed stride in the first convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding) # Stride is 1 here
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Modulation layers: Project unvoiced embedding to gamma and beta for bn2
        # Use Conv1d to potentially capture temporal variations in modulation
        self.mod_conv = nn.Conv1d(unvoiced_embed_dim, out_channels * 2, kernel_size=1) # Output gamma and beta

        # Shortcut connection: Apply stride if stride > 1 OR if channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Use the same stride in the shortcut's convolution
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x, unvoiced_embedding):
        residual = self.shortcut(x)

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (before modulation)
        out = self.conv2(out)
        out = self.bn2(out) # Apply BN before modulation

        # Calculate modulation parameters (gamma, beta) using Conv1d approach
        # unvoiced_embedding shape: (B, U_dim, T)
        mod_params = self.mod_conv(unvoiced_embedding) # Shape: (B, 2 * out_channels, T)
        gamma, beta = torch.chunk(mod_params, 2, dim=1) # Each: (B, out_channels, T)

        # Apply modulation (FiLM-like)
        out = gamma * out + beta

        # Add residual and apply final ReLU
        out += residual
        out = self.relu(out)
        return out

# Modified LowResModel using ResidualBlock1D with modulation
class LowResModel(nn.Module):
    # Updated __init__ for multi-band output projection
    def __init__(self, input_dim, channels_list, output_dim, unvoiced_embed_dim, config=None): # Added config
        super().__init__()
        
        self.reshape = nn.Conv1d(input_dim, channels_list[0], kernel_size=1)
        
        self.blocks = nn.ModuleList()
        current_channels = channels_list[0]
        # Iterate through all target channels to define blocks
        for i in range(len(channels_list)):
            out_channels = channels_list[i] # Target channel for this block
            # Use stride=2 for the first block to downsample time, stride=1 otherwise
            stride = 2 if i == 0 else 1
            self.blocks.append(ResidualBlock1D(current_channels, out_channels, unvoiced_embed_dim, stride=stride))
            current_channels = out_channels # Update current channels for the next block
        
        # --- Multi-band Output Projection ---
        if config is None:
            # Maintain backward compatibility if config is not passed (treat as single band)
            logger.warning("Config not passed to LowResModel. Assuming single band output.")
            self.num_bands = 1
        else:
            self.num_bands = config['model'].get('num_bands_stage1', 1) # Default to 1 band

        if self.num_bands > 1:
            logger.info(f"LowResModel: Using {self.num_bands} bands for output projection.")
            if output_dim % self.num_bands != 0:
                raise ValueError(f"LowResModel output_dim ({output_dim}) must be divisible by num_bands_stage1 ({self.num_bands})")
            self.bins_per_band = output_dim // self.num_bands
            self.band_output_projs = nn.ModuleList()
            for _ in range(self.num_bands):
                self.band_output_projs.append(
                    nn.Conv1d(channels_list[-1], self.bins_per_band, kernel_size=1)
                )
        else: # Single band (original behavior)
            self.output_proj = nn.Conv1d(channels_list[-1], output_dim, kernel_size=1)
        
    def forward(self, x, unvoiced_embedding): # Added unvoiced_embedding
        x = self.reshape(x)
        
        # Downsample unvoiced_embedding if the first block uses stride > 1
        cond_embedding = unvoiced_embedding
        if len(self.blocks) > 0:
            first_block_stride = self.blocks[0].conv1.stride[0] # Get stride value
            if first_block_stride > 1:
                # Use functional avg_pool1d for downsampling
                cond_embedding = F.avg_pool1d(unvoiced_embedding, kernel_size=first_block_stride, stride=first_block_stride)

        # Pass potentially downsampled embedding to each block
        for block in self.blocks:
            x = block(x, cond_embedding)
        
        # Apply output projection(s)
        if self.num_bands > 1:
            band_outputs = []
            for proj_layer in self.band_output_projs:
                band_outputs.append(proj_layer(x))
            # Concatenate along frequency dimension (dim=1)
            x = torch.cat(band_outputs, dim=1) # Shape: [B, F_low, T_downsampled]
        else:
            x = self.output_proj(x) # Shape: [B, F_low, T_downsampled]

        # Permute output from (B, F_low, T_downsampled) to (B, T_downsampled, F_low)
        return x.permute(0, 2, 1)
# New ModulatedConvBlock2D class definition
class ModulatedConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Use Conv1d to generate modulation params per time step from conditioning input
        self.modulation_conv = nn.Conv1d(conditioning_dim, out_channels * 2, kernel_size=1)

    def forward(self, x, cond):
        # x shape: [B, C_in, F, T]
        # cond shape: [B, cond_dim, T]

        # Standard convolution path
        h = self.conv(x)
        h = self.bn(h) # Apply BN before modulation

        # Generate modulation parameters (gamma, beta)
        mod_params = self.modulation_conv(cond) # Shape: [B, 2 * out_channels, T]
        gamma, beta = torch.chunk(mod_params, 2, dim=1) # Each: [B, out_channels, T]

        # Expand gamma and beta for broadcasting with h's shape [B, C_out, F, T]
        # Add frequency dimension: [B, out_channels, 1, T]
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        # Apply modulation (FiLM-like)
        h = gamma * h + beta

        # Apply activation
        h = self.relu(h)
        return h
# Residual Block with Modulation for 2D data (Spectrograms)
class ModulatedResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Ensure padding maintains dimensions for stride=1
        padding = kernel_size // 2 if stride == 1 else padding

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # Bias False common in ResBlocks before BN
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2, bias=False) # Stride 1, padding to maintain size
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Modulation layers (similar to ModulatedConvBlock2D)
        self.modulation_conv = nn.Conv1d(conditioning_dim, out_channels * 2, kernel_size=1)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, cond):
        # x shape: [B, C_in, F, T]
        # cond shape: [B, cond_dim, T]

        residual = self.shortcut(x)

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (before modulation)
        out = self.conv2(out)
        out = self.bn2(out) # Apply BN before modulation

        # Generate modulation parameters (gamma, beta)
        # cond shape: [B, cond_dim, T]
        mod_params = self.modulation_conv(cond) # Shape: [B, 2 * out_channels, T]
        gamma, beta = torch.chunk(mod_params, 2, dim=1) # Each: [B, out_channels, T]

        # Expand gamma and beta for broadcasting: [B, out_channels, 1, T]
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        # Apply modulation (FiLM-like)
        out = gamma * out + beta

        # Add residual and apply final ReLU
        out += residual
        out = self.relu(out)
        return out




# Modify MidResUpsampler
class MidResUpsampler(nn.Module):
    # Updated __init__ to accept embedding dimensions and stride
    # Updated __init__ for multi-band processing
    def __init__(self, input_dim, channels_list, output_dim,
                 f0_embed_dim, phone_embed_dim, midi_embed_dim, unvoiced_embed_dim,
                 downsample_stride=2, config=None): # Added config parameter
        super().__init__()
        if config is None:
             raise ValueError("Config dictionary must be provided to MidResUpsampler")

        # Store stride and calculate total embedding dimension
        self.downsample_stride = downsample_stride
        self.total_embed_dim = f0_embed_dim + phone_embed_dim + midi_embed_dim + unvoiced_embed_dim

        # --- Multi-band Configuration ---
        self.num_bands = config['model'].get('num_bands_stage2', 1) # Default to 1 band if not specified
        self.band_processing = config['model'].get('band_processing_stage2', 'shared') # Default to shared
        logger.info(f"MidResUpsampler: Using {self.num_bands} bands with '{self.band_processing}' processing.")

        # --- Upsampling (Common to all bands) ---
        # Replace Upsample + ConvBlock with ConvTranspose2d for learnable frequency upsampling
        self.upsample_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

        # --- Band Processing Blocks ---
        if self.num_bands > 1 and self.band_processing == 'separate':
            self.band_processors = nn.ModuleList()
            for _ in range(self.num_bands):
                band_block_list = nn.ModuleList()
                # Initial block for this band (input channel = 1)
                band_block_list.append(ModulatedResidualBlock2D(1, channels_list[0], self.total_embed_dim))
                # Subsequent blocks for this band
                for i in range(len(channels_list) - 1):
                    band_block_list.append(ModulatedResidualBlock2D(channels_list[i], channels_list[i+1], self.total_embed_dim))
                self.band_processors.append(band_block_list)
        else: # Shared processing (or single band)
            self.shared_processor = nn.ModuleList()
            # Initial block (input channel = 1)
            self.shared_processor.append(ModulatedResidualBlock2D(1, channels_list[0], self.total_embed_dim))
            # Subsequent blocks
            for i in range(len(channels_list) - 1):
                self.shared_processor.append(ModulatedResidualBlock2D(channels_list[i], channels_list[i+1], self.total_embed_dim))

        # --- Final Projection (Common to all bands after concatenation) ---
        # --- Smoothing Layer (Optional, for separate bands) ---
        self.smoothing_conv = ConvBlock2D(channels_list[-1], channels_list[-1], kernel_size=(3, 1), padding=(1, 0))

        # --- Final Projection (Common to all bands after concatenation) ---
        # Input channels = last channel size in channels_list
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    # Updated forward for multi-band processing
    def forward(self, x, f0_enc_orig, phone_enc_orig, midi_enc_orig, unvoiced_enc_orig):
        # x shape: [B, T_downsampled, F_in] (Output from LowResModel)
        # Embeddings shape: [B, T_orig, Dim]

        # --- Process Spectrogram (Initial Upsampling) ---
        # Permute x to [B, F_in, T_downsampled]
        x = x.permute(0, 2, 1)
        # Unsqueeze x to [B, 1, F_in, T_downsampled] for 2D operations
        x = x.unsqueeze(1)
        # Upsample Frequency using ConvTranspose2d: [B, 1, F_in, T] -> [B, 1, F_mid, T]
        x_upsampled = self.upsample_conv(x) # Shape: [B, 1, F_mid, T]

        # --- Process Embeddings (Common Conditioning) ---
        # Concatenate original embeddings along feature dim: [B, T_orig, TotalDim]
        all_embeddings_orig = torch.cat((f0_enc_orig, phone_enc_orig, midi_enc_orig, unvoiced_enc_orig), dim=2)
        # Permute for 1D pooling/conv: [B, TotalDim, T_orig]
        all_embeddings_orig_permuted = all_embeddings_orig.permute(0, 2, 1)
        # Downsample time dimension if needed: [B, TotalDim, T_downsampled]
        if self.downsample_stride > 1:
            # Determine target time dimension from the input spectrogram x [B, T_downsampled, F_in]
            target_time_dim = x.shape[1]
            all_embeddings_downsampled = F.interpolate(all_embeddings_orig_permuted,
                                                       size=target_time_dim,
                                                       mode='linear',
                                                       align_corners=False) # align_corners=False recommended for linear
        else:
            all_embeddings_downsampled = all_embeddings_orig_permuted
        # cond_signal shape: [B, TotalDim, T_downsampled]
        cond_signal = all_embeddings_downsampled

        # --- Multi-Band Processing ---
        if self.num_bands > 1:
            # Squeeze channel dim before splitting: [B, F_mid, T]
            x_squeezed = x_upsampled.squeeze(1)
            # Split into bands along frequency dim (dim=1)
            # Note: torch.chunk might create uneven chunks if F_mid is not divisible by num_bands
            band_tensors = torch.chunk(x_squeezed, self.num_bands, dim=1)

            processed_bands = []
            for i, band_tensor in enumerate(band_tensors):
                # Unsqueeze channel dim for processing: [B, 1, F_band, T]
                band_input = band_tensor.unsqueeze(1)

                if self.band_processing == 'separate':
                    # Ensure band_processors exists (created in __init__)
                    if not hasattr(self, 'band_processors'):
                         raise AttributeError("MidResUpsampler configured for separate bands, but band_processors not initialized.")
                    processor = self.band_processors[i]
                else: # Shared processor
                    # Ensure shared_processor exists
                    if not hasattr(self, 'shared_processor'):
                         raise AttributeError("MidResUpsampler configured for shared bands, but shared_processor not initialized.")
                    processor = self.shared_processor

                # Process the band
                processed_band = band_input
                for block in processor:
                    processed_band = block(processed_band, cond_signal)
                # processed_band shape: [B, C_last, F_band, T]
                processed_bands.append(processed_band)

            # Concatenate processed bands along frequency dim (dim=2)
            x_processed = torch.cat(processed_bands, dim=2) # Shape: [B, C_last, F_mid, T]

            # --- Apply Smoothing if Separate Bands ---
            if self.band_processing == 'separate':
                 # Ensure smoothing_conv exists (defined in __init__)
                if not hasattr(self, 'smoothing_conv'):
                     raise AttributeError("MidResUpsampler configured for separate bands, but smoothing_conv not initialized.")
                x_processed = self.smoothing_conv(x_processed) # Apply smoothing

        else: # Single band processing (original path, but using shared_processor)
             # Ensure shared_processor exists
            if not hasattr(self, 'shared_processor'):
                 raise AttributeError("MidResUpsampler configured for single band, but shared_processor not initialized.")
            processor = self.shared_processor
            x_processed = x_upsampled # Start with [B, 1, F_mid, T]
            for block in processor:
                x_processed = block(x_processed, cond_signal)
            # x_processed shape: [B, C_last, F_mid, T]

        # --- Final Projection ---
        # Output projection: [B, C_last, F_mid, T] -> [B, 1, F_mid, T]
        x = self.output_proj(x_processed)

        # Squeeze channel dim: [B, 1, F_mid, T] -> [B, F_mid, T]
        x = x.squeeze(1)
        # Permute back to [B, T, F_mid]
        return x.permute(0, 2, 1)

class HighResUpsampler(nn.Module):
    # Updated __init__ to accept embedding dimensions and stride
    # Updated __init__ for multi-band processing
    def __init__(self, input_dim, channels_list, output_dim,
                 f0_embed_dim, phone_embed_dim, midi_embed_dim, unvoiced_embed_dim,
                 downsample_stride=2, config=None): # Added config parameter
        super().__init__()
        if config is None:
             raise ValueError("Config dictionary must be provided to HighResUpsampler")

        # Store stride and calculate total embedding dimension
        self.downsample_stride = downsample_stride
        self.total_embed_dim = f0_embed_dim + phone_embed_dim + midi_embed_dim + unvoiced_embed_dim

        # --- Multi-band Configuration ---
        self.num_bands = config['model'].get('num_bands_stage3', 1) # Default to 1 band
        self.band_processing = config['model'].get('band_processing_stage3', 'shared') # Default to shared
        logger.info(f"HighResUpsampler: Using {self.num_bands} bands with '{self.band_processing}' processing.")

        # --- Upsampling (Common to all bands) ---
        # Keep original ConvTranspose2d for frequency upsampling in Stage 3
        self.upsample_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

        # --- Band Processing Blocks ---
        if self.num_bands > 1 and self.band_processing == 'separate':
            self.band_processors = nn.ModuleList()
            for _ in range(self.num_bands):
                band_block_list = nn.ModuleList()
                # Initial block for this band (input channel = 1 after upsample_conv)
                band_block_list.append(ModulatedResidualBlock2D(1, channels_list[0], self.total_embed_dim))
                # Subsequent blocks for this band
                for i in range(len(channels_list) - 1):
                    band_block_list.append(ModulatedResidualBlock2D(channels_list[i], channels_list[i+1], self.total_embed_dim))
                self.band_processors.append(band_block_list)
        else: # Shared processing (or single band)
            self.shared_processor = nn.ModuleList()
            # Initial block (input channel = 1 after upsample_conv)
            self.shared_processor.append(ModulatedResidualBlock2D(1, channels_list[0], self.total_embed_dim))
            # Subsequent blocks
            for i in range(len(channels_list) - 1):
                self.shared_processor.append(ModulatedResidualBlock2D(channels_list[i], channels_list[i+1], self.total_embed_dim))

        # --- Smoothing Layer (Optional, for separate bands) ---
        self.smoothing_conv = ConvBlock2D(channels_list[-1], channels_list[-1], kernel_size=(3, 1), padding=(1, 0))

        # --- Final Projection (Common to all bands after concatenation) ---
        # Input channels = last channel size in channels_list
        self.output_proj = nn.Conv2d(channels_list[-1], 1, kernel_size=1)
        
    # Updated forward for multi-band processing
    def forward(self, x, f0_enc_orig, phone_enc_orig, midi_enc_orig, unvoiced_enc_orig):
        # x shape: [B, T_downsampled, F_mid] (Output from MidResUpsampler)
        # Embeddings shape: [B, T_orig, Dim]

        # --- Process Spectrogram (Initial Upsampling) ---
        # Permute x to [B, F_mid, T_downsampled]
        x = x.permute(0, 2, 1)
        # Unsqueeze x to [B, 1, F_mid, T_downsampled] for 2D operations
        x = x.unsqueeze(1)
        # Upsample Frequency using ConvTranspose2d: [B, 1, F_mid, T] -> [B, 1, F_high, T]
        x_upsampled = self.upsample_conv(x) # Shape: [B, 1, F_high, T]

        # --- Process Embeddings (Common Conditioning) ---
        # Concatenate original embeddings along feature dim: [B, T_orig, TotalDim]
        all_embeddings_orig = torch.cat((f0_enc_orig, phone_enc_orig, midi_enc_orig, unvoiced_enc_orig), dim=2)
        # Permute for 1D pooling/conv: [B, TotalDim, T_orig]
        all_embeddings_orig_permuted = all_embeddings_orig.permute(0, 2, 1)
        # Downsample time dimension if needed: [B, TotalDim, T_downsampled]
        if self.downsample_stride > 1:
            # Determine target time dimension from the input spectrogram x [B, T_downsampled, F_mid]
            target_time_dim = x.shape[1]
            all_embeddings_downsampled = F.interpolate(all_embeddings_orig_permuted,
                                                       size=target_time_dim,
                                                       mode='linear',
                                                       align_corners=False) # align_corners=False recommended for linear
        else:
            all_embeddings_downsampled = all_embeddings_orig_permuted
        # cond_signal shape: [B, TotalDim, T_downsampled]
        cond_signal = all_embeddings_downsampled

        # --- Multi-Band Processing ---
        if self.num_bands > 1:
            # Squeeze channel dim before splitting: [B, F_high, T]
            x_squeezed = x_upsampled.squeeze(1)
            # Split into bands along frequency dim (dim=1)
            band_tensors = torch.chunk(x_squeezed, self.num_bands, dim=1)

            processed_bands = []
            for i, band_tensor in enumerate(band_tensors):
                # Unsqueeze channel dim for processing: [B, 1, F_band, T]
                band_input = band_tensor.unsqueeze(1)

                if self.band_processing == 'separate':
                    if not hasattr(self, 'band_processors'):
                         raise AttributeError("HighResUpsampler configured for separate bands, but band_processors not initialized.")
                    processor = self.band_processors[i]
                else: # Shared processor
                    if not hasattr(self, 'shared_processor'):
                         raise AttributeError("HighResUpsampler configured for shared bands, but shared_processor not initialized.")
                    processor = self.shared_processor

                # Process the band
                processed_band = band_input
                for block in processor:
                    processed_band = block(processed_band, cond_signal)
                # processed_band shape: [B, C_last, F_band, T]
                processed_bands.append(processed_band)

            # Concatenate processed bands along frequency dim (dim=2)
            x_processed = torch.cat(processed_bands, dim=2) # Shape: [B, C_last, F_high, T]

            # --- Apply Smoothing if Separate Bands ---
            if self.band_processing == 'separate':
                if not hasattr(self, 'smoothing_conv'):
                     raise AttributeError("HighResUpsampler configured for separate bands, but smoothing_conv not initialized.")
                x_processed = self.smoothing_conv(x_processed) # Apply smoothing

        else: # Single band processing (original path, but using shared_processor)
            if not hasattr(self, 'shared_processor'):
                 raise AttributeError("HighResUpsampler configured for single band, but shared_processor not initialized.")
            processor = self.shared_processor
            x_processed = x_upsampled # Start with [B, 1, F_high, T]
            for block in processor:
                x_processed = block(x_processed, cond_signal)
            # x_processed shape: [B, C_last, F_high, T]

        # --- Final Projection ---
        # Output projection: [B, C_last, F_high, T] -> [B, 1, F_high, T]
        x = self.output_proj(x_processed)

        # Squeeze channel dim: [B, 1, F_high, T] -> [B, F_high, T]
        x = x.squeeze(1)
        # Permute back to [B, T, F_high]
        return x.permute(0, 2, 1)