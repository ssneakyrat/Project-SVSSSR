# models/progressive_svs.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision
from models.blocks import LowResModel, MidResUpsampler, HighResUpsampler
from utils.plotting import plot_spectrograms_to_figure # Import the plotting utility


class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.f0_embed_dim = config['model']['f0_embed_dim']
        self.phone_embed_dim = config['model']['phone_embed_dim']
        self.midi_embed_dim = config['model']['midi_embed_dim']
        self.unvoiced_embed_dim = config['model']['unvoiced_embed_dim'] # Read from config

        # F0 encoding (continuous value)
        self.f0_projection = nn.Linear(1, self.f0_embed_dim)

        # Phone encoding (categorical)
        num_phones = len(config.get('phone_map', []))
        if num_phones == 0:
            num_phones = 100  # Default if not specified
        self.phone_embedding = nn.Embedding(num_phones, self.phone_embed_dim)

        # MIDI encoding (categorical)
        num_midi_notes = 128  # MIDI standard
        self.midi_embedding = nn.Embedding(num_midi_notes, self.midi_embed_dim)

        # Unvoiced flag encoding (continuous value 0.0 or 1.0)
        self.unvoiced_projection = nn.Linear(1, self.unvoiced_embed_dim)

    def forward(self, f0, phone_label, phone_duration, midi_label, unvoiced_flag):
        # f0 has shape (B, T, 1), get B and T correctly
        batch_size = f0.size(0)
        seq_len = f0.size(1)

        # Encode F0 (frame-level)
        f0_encoded = self.f0_projection(f0)  # [B, T, F0_dim]

        # Encode phone (frame-level)
        phone_encoded = self.phone_embedding(phone_label.long())  # [B, T, P_dim]

        # Encode MIDI (frame-level)
        midi_label_squeezed = midi_label.long()
        if midi_label_squeezed.dim() == 3 and midi_label_squeezed.size(2) == 1:
             midi_label_squeezed = midi_label_squeezed.squeeze(2) # Shape becomes (B, T)
        midi_encoded = self.midi_embedding(midi_label_squeezed)  # [B, T, M_dim]

        # Encode unvoiced flag (frame-level) separately
        if unvoiced_flag.dim() == 2:
            unvoiced_flag = unvoiced_flag.unsqueeze(-1)
        unvoiced_encoded = self.unvoiced_projection(unvoiced_flag) # [B, T, U_dim]

        # Return individual embeddings before concatenation/transposition
        # Shapes: [B, T, Dim]
        return f0_encoded, phone_encoded, midi_encoded, unvoiced_encoded

class ProgressiveSVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters('config')

        self.config = config
        self.current_stage = config['model'].get('current_stage', 1)

        # Feature encoder
        self.feature_encoder = FeatureEncoder(config)
        # Store individual embedding dims for later use
        self.f0_embed_dim = self.feature_encoder.f0_embed_dim
        self.phone_embed_dim = self.feature_encoder.phone_embed_dim
        self.midi_embed_dim = self.feature_encoder.midi_embed_dim
        self.unvoiced_embed_dim = self.feature_encoder.unvoiced_embed_dim

        # Progressive stages
        # Input dim for LowResModel (still uses concatenated main features)
        input_dim_main = self.f0_embed_dim + self.phone_embed_dim + self.midi_embed_dim
        # We stored individual dims earlier (self.f0_embed_dim, etc.)

        # Calculate frequency dimensions for each stage
        full_mel_bins = config['model']['mel_bins']
        low_res_freq_bins = int(full_mel_bins / config['model']['low_res_scale'])
        mid_res_freq_bins = int(full_mel_bins / config['model']['mid_res_scale'])

        # Stage 1: Low-Resolution Model
        # Determine the stride used in the first block of LowResModel for downsampling
        # Assuming stride=2 for the first block as per LowResModel implementation
        self.downsample_stride = 2 if len(config['model']['low_res_channels']) > 0 else 1

        self.low_res_model = LowResModel(
            input_dim=input_dim_main,
            channels_list=config['model']['low_res_channels'],
            output_dim=low_res_freq_bins,
            unvoiced_embed_dim=self.unvoiced_embed_dim, # Pass only unvoiced dim here
            config=config # Pass the config dictionary
        )

        # Stage 2: Mid-Resolution Upsampler (40×432)
        # Stage 2: Mid-Resolution Upsampler - Pass all embedding dims and stride
        self.mid_res_upsampler = MidResUpsampler(
            input_dim=low_res_freq_bins, # Freq dim from previous stage
            channels_list=config['model']['mid_res_channels'],
            output_dim=mid_res_freq_bins,
            f0_embed_dim=self.f0_embed_dim,
            phone_embed_dim=self.phone_embed_dim,
            midi_embed_dim=self.midi_embed_dim,
            unvoiced_embed_dim=self.unvoiced_embed_dim,
            downsample_stride=self.downsample_stride,
            config=config # Pass the config dictionary
        )

        # Stage 3: High-Resolution Upsampler (80×864)
        # Stage 3: High-Resolution Upsampler - Pass all embedding dims and stride
        self.high_res_upsampler = HighResUpsampler(
            input_dim=mid_res_freq_bins, # Freq dim from previous stage
            channels_list=config['model']['high_res_channels'], # Use updated channels
            output_dim=full_mel_bins,
            f0_embed_dim=self.f0_embed_dim,
            phone_embed_dim=self.phone_embed_dim,
            midi_embed_dim=self.midi_embed_dim,
            unvoiced_embed_dim=self.unvoiced_embed_dim,
            downsample_stride=self.downsample_stride, # Same stride as MidRes
            config=config # Pass the config dictionary
        )

        # Loss function
        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, f0, phone_label, phone_duration, midi_label, unvoiced_flag):
        # Encode input features - Returns individual embeddings [B, T_orig, Dim]
        f0_enc, phone_enc, midi_enc, unvoiced_enc = self.feature_encoder(
            f0, phone_label, phone_duration, midi_label, unvoiced_flag
        )

        # --- Prepare inputs for Stage 1 ---
        # Concatenate main features (F0, phone, MIDI) [B, T_orig, F0+P+M_dim]
        main_features_cat = torch.cat([f0_enc, phone_enc, midi_enc], dim=2)
        # Transpose for Conv1D: [B, F0+P+M_dim, T_orig]
        main_features_s1 = main_features_cat.transpose(1, 2)
        # Transpose unvoiced embedding for Conv1D: [B, U_dim, T_orig]
        unvoiced_embedding_s1 = unvoiced_enc.transpose(1, 2)

        # --- Stage 1: Low-Resolution Model ---
        # Input: Transposed main features and unvoiced embedding
        # Output: Low-res spectrogram [B, T_downsampled, F_low]
        low_res_output = self.low_res_model(main_features_s1, unvoiced_embedding_s1)


        if self.current_stage == 1:
            return low_res_output

        # --- Stage 2: Mid-Resolution Upsampler ---
        # Input: Low-res spec and ORIGINAL embeddings
        mid_res_output = self.mid_res_upsampler(
            low_res_output, f0_enc, phone_enc, midi_enc, unvoiced_enc
        )

        if self.current_stage == 2:
            return mid_res_output

        # --- Stage 3: High-Resolution Upsampler ---
        # Input: Mid-res spec and ORIGINAL embeddings
        high_res_output = self.high_res_upsampler(
            mid_res_output, f0_enc, phone_enc, midi_enc, unvoiced_enc
        )


        return high_res_output

    def training_step(self, batch, batch_idx):
        mel_specs = batch['mel_spec']
        f0 = batch['f0']
        phone_label = batch['phone_label']
        phone_duration = batch['phone_duration']
        midi_label = batch['midi_label']
        lengths = batch['length']
        voiced_mask = batch['voiced_mask'] # Get the voiced mask
        unvoiced_flag = batch['unvoiced_flag'] # Get the unvoiced flag

        # Forward pass
        mel_pred = self(f0, phone_label, phone_duration, midi_label, unvoiced_flag)

        # Create masks for variable length based on ORIGINAL time dimension
        original_max_len = mel_specs.shape[1] # Use time dimension (dim 1), e.g., 862
        original_mask = torch.arange(original_max_len, device=lengths.device).expand(len(lengths), original_max_len) < lengths.unsqueeze(1)
        # Target time dimension after downsampling
        target_time_dim = original_max_len // self.downsample_stride # Use stored stride

        # Adjust target resolution based on current stage
        freq_weights_tensor = None # Initialize frequency weights
        if self.current_stage == 1:
            # Downsample ground truth for low-res stage
            scale_factor = 1/self.config['model']['low_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)

            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # t is original time dim (e.g., 862)
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Downsample BOTH frequency and time
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, target_time_dim), # Target shape (Freq', Time/stride)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T/stride)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)

            # Downsample original_mask to match target_time_dim (T/stride)
            mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            downsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time/stride)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T/stride)
            # Expand mask to match mel_target shape (B, T/stride, F')
            mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T/stride, F')
        elif self.current_stage == 2:
            # Downsample ground truth for mid-res stage
            scale_factor = 1/self.config['model']['mid_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)

            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # t is original time dim (e.g., 862)
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Downsample BOTH frequency and time (target_time_dim = t // stride)
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, target_time_dim), # Target shape (Freq', Time/stride)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T/stride)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)

            # Downsample original_mask to match target_time_dim (T/stride)
            mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            downsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time/stride)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T/stride)
            # Expand mask to match mel_target shape (B, T/stride, F')
            mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T/stride, F')
            # --- Define Frequency Weights for Stage 2 ---
            freq_weights = torch.linspace(1.0, 2.0, freq_dim, device=self.device)
            # Reshape for broadcasting: [F] -> [1, 1, F]
            freq_weights_tensor = freq_weights.unsqueeze(0).unsqueeze(0)
            # --- End Frequency Weights ---
        else: # Stage 3
            # Stage 3: Full frequency resolution, but DOWNsampled time
            b, t, c = mel_specs.shape # t is original time dim (e.g., 862)
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            mel_target = F.interpolate(
                mel_specs_4d,
                size=(c, target_time_dim), # Target shape (Freq_full, Time/stride)
                mode='bilinear',
                align_corners=False
            ).squeeze(1) # Back to (B, F_full, T/stride)
            mel_target = mel_target.permute(0, 2, 1) # Back to (B, T/stride, F_full)

            # Downsample original_mask to match target_time_dim (T/stride)
            mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            downsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time/stride)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T/stride)
            # Expand mask to match mel_target shape (B, T/stride, F_full)
            mask = downsampled_mask.unsqueeze(2).expand(-1, -1, c) # Expand to (B, T/stride, F_full)
            # --- Define Frequency Weights for Stage 3 ---
            # Note: c is the full frequency dimension here
            freq_weights = torch.linspace(1.0, 2.0, c, device=self.device)
            # Reshape for broadcasting: [F] -> [1, 1, F]
            freq_weights_tensor = freq_weights.unsqueeze(0).unsqueeze(0)
            # --- End Frequency Weights ---

        # Handle dimensionality issues if mel_pred shape is unexpected
        if mel_pred.dim() != mel_target.dim():
             # Attempt to fix common shape mismatches, log a warning
             print(f"Warning: Shape mismatch detected in training_step. Pred: {mel_pred.shape}, Target: {mel_target.shape}. Attempting correction.")
             if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
                 mel_pred = mel_pred.squeeze(1) # [B, 1, F, T] -> [B, F, T]
                 mel_pred = mel_pred.permute(0, 2, 1) # [B, F, T] -> [B, T, F] to match target
             elif mel_pred.dim() == 2 and mel_target.dim() == 3:
                 mel_pred = mel_pred.unsqueeze(0) # Add batch dim if missing
             # Add more specific corrections if needed based on observed errors

        # Ensure shapes match before loss calculation
        if mel_pred.shape != mel_target.shape:
            print(f"ERROR: Unresolvable shape mismatch in training_step. Pred: {mel_pred.shape}, Target: {mel_target.shape}. Skipping loss calculation.")
            # Return a dummy loss or raise an error
            return torch.tensor(0.0, device=self.device, requires_grad=True)


        # Compute element-wise loss
        loss_elementwise = self.loss_fn(mel_pred, mel_target)

        # --- Apply Weighted Loss ---
        unvoiced_weight = 2.0 # Weight for unvoiced frames
        # Ensure voiced_mask has shape (B, T) before unsqueezing
        if voiced_mask.dim() == 3 and voiced_mask.shape[-1] == 1:
             voiced_mask = voiced_mask.squeeze(-1) # Make it (B, T_original)

        # Downsample voiced_mask to match target time dimension (T/stride)
        voiced_mask_float = voiced_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T_original)
        downsampled_voiced_mask = F.interpolate(
            voiced_mask_float,
            size=(1, target_time_dim), # Target shape (1, Time/stride)
            mode='nearest' # Use nearest to keep boolean-like nature
        ).squeeze(1).squeeze(1) # Back to (B, T/stride)
        downsampled_voiced_mask = downsampled_voiced_mask > 0.5 # Convert back to boolean-like (0. or 1.)

        # Create weights: shape (B, T/stride, 1), broadcastable to (B, T/stride, F)
        loss_weights = torch.where(downsampled_voiced_mask.unsqueeze(-1), 1.0, unvoiced_weight)
        loss_weights = loss_weights.to(loss_elementwise.device) # Ensure same device
        # Apply length mask and loss weights
        # mask has shape (B, T/stride, F)
        # Apply frequency weights if defined (for stages 2 and 3)
        if freq_weights_tensor is not None:
            weighted_loss = loss_elementwise * mask.float() * loss_weights * freq_weights_tensor
            total_weight_sum = (mask.float() * loss_weights * freq_weights_tensor).sum()
        else: # Stage 1 (no frequency weighting)
            weighted_loss = loss_elementwise * mask.float() * loss_weights
            total_weight_sum = (mask.float() * loss_weights).sum()

        loss = weighted_loss.sum() / total_weight_sum.clamp(min=1e-8)
        # --- End Weighted Loss ---

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel_specs = batch['mel_spec']
        f0 = batch['f0']
        phone_label = batch['phone_label']
        phone_duration = batch['phone_duration']
        midi_label = batch['midi_label']
        lengths = batch['length']
        voiced_mask = batch['voiced_mask'] # Get the voiced mask
        unvoiced_flag = batch['unvoiced_flag'] # Get the unvoiced flag

        # Forward pass
        mel_pred = self(f0, phone_label, phone_duration, midi_label, unvoiced_flag)

        # Create masks for variable length based on ORIGINAL time dimension
        original_max_len = mel_specs.shape[1] # Use time dimension (dim 1), e.g., 862
        original_mask = torch.arange(original_max_len, device=lengths.device).expand(len(lengths), original_max_len) < lengths.unsqueeze(1)
        # Target time dimension after downsampling
        target_time_dim = original_max_len // self.downsample_stride # Use stored stride

        # Adjust target resolution based on current stage
        freq_weights_tensor = None # Initialize frequency weights
        if self.current_stage == 1:
            # Downsample ground truth for low-res stage
            scale_factor = 1/self.config['model']['low_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)

            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # t is original time dim (e.g., 862)
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Downsample BOTH frequency and time
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, target_time_dim), # Target shape (Freq', Time/stride)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T/stride)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)
            # Downsample original_mask to match target_time_dim (T/stride)
            mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            downsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time/stride)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T/stride)
            # Expand mask to match mel_target shape (B, T/stride, F')
            mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T/stride, F')

        elif self.current_stage == 2:
            # Downsample ground truth for mid-res stage
            scale_factor = 1/self.config['model']['mid_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)

            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # t is original time dim (e.g., 862)
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Downsample BOTH frequency and time (target_time_dim = t // stride)
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, target_time_dim), # Target shape (Freq', Time/stride)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T/stride)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)
            # Downsample original_mask to match target_time_dim (T/stride)
            mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            downsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time/stride)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T/stride)
            # Expand mask to match mel_target shape (B, T/stride, F')
            mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T/stride, F')
            # --- Define Frequency Weights for Stage 2 ---
            freq_weights = torch.linspace(1.0, 2.0, freq_dim, device=self.device)
            # Reshape for broadcasting: [F] -> [1, 1, F]
            freq_weights_tensor = freq_weights.unsqueeze(0).unsqueeze(0)
            # --- End Frequency Weights ---

        else: # Stage 3
            # Stage 3: Full frequency resolution, but DOWNsampled time
            b, t, c = mel_specs.shape # t is original time dim (e.g., 862)
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            mel_target = F.interpolate(
                mel_specs_4d,
                size=(c, target_time_dim), # Target shape (Freq_full, Time/stride)
                mode='bilinear',
                align_corners=False
            ).squeeze(1) # Back to (B, F_full, T/stride)
            mel_target = mel_target.permute(0, 2, 1) # Back to (B, T/stride, F_full)

            # Downsample original_mask to match target_time_dim (T/stride)
            mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            downsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time/stride)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T/stride)
            # Expand mask to match mel_target shape (B, T/stride, F_full)
            mask = downsampled_mask.unsqueeze(2).expand(-1, -1, c) # Expand to (B, T/stride, F_full)
            # --- Define Frequency Weights for Stage 3 ---
            # Note: c is the full frequency dimension here
            freq_weights = torch.linspace(1.0, 2.0, c, device=self.device)
            # Reshape for broadcasting: [F] -> [1, 1, F]
            freq_weights_tensor = freq_weights.unsqueeze(0).unsqueeze(0)
            # --- End Frequency Weights ---

        # Handle dimensionality issues if mel_pred shape is unexpected
        if mel_pred.dim() != mel_target.dim():
             # Attempt to fix common shape mismatches, log a warning
             print(f"Warning: Shape mismatch detected in validation_step. Pred: {mel_pred.shape}, Target: {mel_target.shape}. Attempting correction.")
             if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
                 mel_pred = mel_pred.squeeze(1) # [B, 1, F, T] -> [B, F, T]
                 mel_pred = mel_pred.permute(0, 2, 1) # [B, F, T] -> [B, T, F] to match target
             elif mel_pred.dim() == 2 and mel_target.dim() == 3:
                 mel_pred = mel_pred.unsqueeze(0) # Add batch dim if missing
             # Add more specific corrections if needed based on observed errors

        # Ensure shapes match before loss calculation
        if mel_pred.shape != mel_target.shape:
            print(f"ERROR: Unresolvable shape mismatch in validation_step. Pred: {mel_pred.shape}, Target: {mel_target.shape}. Skipping loss calculation.")
            # Log a dummy loss or handle appropriately
            self.log('val_loss', 1e9, prog_bar=True) # Log a high loss value
            return torch.tensor(1e9, device=self.device)


        # Compute element-wise loss
        loss_elementwise = self.loss_fn(mel_pred, mel_target)

        # --- Apply Weighted Loss ---
        unvoiced_weight = 2.0 # Weight for unvoiced frames (same as training)
        # Ensure voiced_mask has shape (B, T) before unsqueezing
        if voiced_mask.dim() == 3 and voiced_mask.shape[-1] == 1:
             voiced_mask = voiced_mask.squeeze(-1) # Make it (B, T_original)

        # Downsample voiced_mask to match target time dimension (T/stride)
        voiced_mask_float = voiced_mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T_original)
        downsampled_voiced_mask = F.interpolate(
            voiced_mask_float,
            size=(1, target_time_dim), # Target shape (1, Time/stride)
            mode='nearest' # Use nearest to keep boolean-like nature
        ).squeeze(1).squeeze(1) # Back to (B, T/stride)
        downsampled_voiced_mask = downsampled_voiced_mask > 0.5 # Convert back to boolean-like (0. or 1.)

        # Create weights: shape (B, T/stride, 1), broadcastable to (B, T/stride, F)
        loss_weights = torch.where(downsampled_voiced_mask.unsqueeze(-1), 1.0, unvoiced_weight)
        loss_weights = loss_weights.to(loss_elementwise.device) # Ensure same device

        # Apply length mask and loss weights
        # mask has shape (B, T/stride, F)
        # Apply frequency weights if defined (for stages 2 and 3)
        if freq_weights_tensor is not None:
            weighted_loss = loss_elementwise * mask.float() * loss_weights * freq_weights_tensor
            total_weight_sum = (mask.float() * loss_weights * freq_weights_tensor).sum()
        else: # Stage 1 (no frequency weighting)
            weighted_loss = loss_elementwise * mask.float() * loss_weights
            total_weight_sum = (mask.float() * loss_weights).sum()

        loss = weighted_loss.sum() / total_weight_sum.clamp(min=1e-8)
        # --- End Weighted Loss ---
        self.log('val_loss', loss, prog_bar=True)

        # Visualize first sample in batch at appropriate intervals
        log_interval = self.config['train'].get('log_spectrogram_every_n_val_epochs', 5) # Use config or default
        if batch_idx == 0 and self.current_epoch % log_interval == 0:
            # Get the actual length
            length = lengths[0].item() # Original length, e.g., 862
            # Adjust length for downsampled time dimension
            downsampled_length = length // self.downsample_stride # e.g., 431
            self._log_mel_comparison(
                mel_pred[0, :downsampled_length, :].detach().cpu(),
                mel_target[0, :downsampled_length, :].detach().cpu()
            )

        return loss

    def _log_mel_comparison(self, pred_mel, target_mel):
        # Calculate vmin and vmax based *only* on the ground truth for consistent scaling per sample
        vmin = target_mel.min()
        vmax = target_mel.max()

        # Transpose tensors from (Time, Freq) to (Freq, Time) for the plotting function
        pred_mel_t = pred_mel.T
        target_mel_t = target_mel.T

        # Use the utility function to create the plot
        fig = plot_spectrograms_to_figure(
            ground_truth=target_mel_t,
            prediction=pred_mel_t,
            title=f'Mel Comparison (Stage {self.current_stage})',
            vmin=vmin,
            vmax=vmax
        )

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig) # Close the figure to free memory

        # Convert to tensor
        img = Image.open(buf)
        img_tensor = torchvision.transforms.ToTensor()(img)

        # Log to TensorBoard
        # Ensure logger exists and is valid before logging
        if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_image'):
             self.logger.experiment.add_image(
                 f'Stage_{self.current_stage}/Mel_Comparison',
                 img_tensor,
                 global_step=self.global_step
             )
        else:
             print("Warning: Logger not available or invalid for image logging.")
        buf.close()


    def configure_optimizers(self):
        # Get stage-specific learning rate
        stage_key = f'stage{self.current_stage}'
        try:
            lr = self.config['train']['learning_rate_per_stage'][stage_key]
            print(f"Configuring optimizer for stage {self.current_stage} with LR: {lr}")
        except KeyError:
            print(f"Warning: Learning rate for '{stage_key}' not found in config. Using default 0.001.")
            lr = 0.001

        # Filter parameters: only optimize those with requires_grad=True
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        # Convert filter object to list for AdamW (some versions might require it)
        trainable_params_list = list(trainable_params)

        if not trainable_params_list:
             print("Warning: No trainable parameters found for the optimizer. Check requires_grad flags.")
             # Return None or handle as appropriate if no params are trainable
             # For now, let's create an optimizer with an empty list, though it won't do anything.
             # Consider raising an error if this state is unexpected.
             optimizer = torch.optim.AdamW([], lr=lr) # Create optimizer with empty param list
             return optimizer # Return only the optimizer if no params are trainable


        optimizer = torch.optim.AdamW(
            trainable_params_list, # Pass the list of trainable parameters
            lr=lr,
            weight_decay=self.config['train'].get('weight_decay', 0.0001)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['train'].get('lr_factor', 0.5),
            patience=self.config['train'].get('lr_patience', 10),
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss', # Monitor validation loss
                'interval': 'epoch',
                'frequency': 1,
            }
        }
