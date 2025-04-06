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
        self.unvoiced_embed_dim = 16 # Define dimension for unvoiced flag embedding
        
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

        # Concatenate main features (F0, phone, MIDI)
        main_features_cat = torch.cat([f0_encoded, phone_encoded, midi_encoded], dim=2)
        
        # Reshape main features and unvoiced embedding for 1D convolution (B, C, T)
        main_features = main_features_cat.transpose(1, 2) # Shape: [B, F0+P+M_dim, T]
        unvoiced_embedding = unvoiced_encoded.transpose(1, 2) # Shape: [B, U_dim, T]
        
        return main_features, unvoiced_embedding

class ProgressiveSVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.current_stage = config['model'].get('current_stage', 1)
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(config)
        
        # Progressive stages
        # Input dim for LowResModel is now only the main features (F0, phone, MIDI)
        input_dim_main = self.feature_encoder.f0_embed_dim + self.feature_encoder.phone_embed_dim + self.feature_encoder.midi_embed_dim
        unvoiced_embed_dim = self.feature_encoder.unvoiced_embed_dim # Get unvoiced dim separately
        
        # Calculate frequency dimensions for each stage
        full_mel_bins = config['model']['mel_bins']
        low_res_freq_bins = int(full_mel_bins / config['model']['low_res_scale'])
        mid_res_freq_bins = int(full_mel_bins / config['model']['mid_res_scale'])
        
        # Stage 1: Low-Resolution Model (20×216) - Pass unvoiced_embed_dim
        self.low_res_model = LowResModel(
            input_dim=input_dim_main,
            channels_list=config['model']['low_res_channels'],
            output_dim=low_res_freq_bins,
            unvoiced_embed_dim=unvoiced_embed_dim # Pass the dimension here
        )
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        self.mid_res_upsampler = MidResUpsampler(
            low_res_freq_bins,  # Input is output of previous stage
            config['model']['mid_res_channels'], 
            output_dim=mid_res_freq_bins
        )
        
        # Stage 3: High-Resolution Upsampler (80×864)
        self.high_res_upsampler = HighResUpsampler(
            mid_res_freq_bins,  # Input is output of previous stage
            config['model']['high_res_channels'], 
            output_dim=full_mel_bins
        )
        
        # Loss function
        self.loss_fn = nn.L1Loss(reduction='none')
        
    def forward(self, f0, phone_label, phone_duration, midi_label, unvoiced_flag):
        # Encode input features - Now returns main_features and unvoiced_embedding
        main_features, unvoiced_embedding = self.feature_encoder(f0, phone_label, phone_duration, midi_label, unvoiced_flag)
        
        
        # Stage 1: Low-Resolution Model (20×216) - Pass both feature sets
        low_res_output = self.low_res_model(main_features, unvoiced_embedding)
        
        
        if self.current_stage == 1:
            return low_res_output
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        mid_res_output = self.mid_res_upsampler(low_res_output)
        
        if self.current_stage == 2:
            return mid_res_output
        
        # Stage 3: High-Resolution Upsampler (80×864)
        high_res_output = self.high_res_upsampler(mid_res_output)
        
        
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
        
        # Create masks for variable length
        max_len = mel_specs.shape[1] # Use time dimension (dim 1)
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        
        # Adjust target resolution based on current stage
        if self.current_stage == 1:
            # Downsample ground truth for low-res stage
            scale_factor = 1/self.config['model']['low_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)
            
            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # Correct unpacking
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Downsample keeping original time dimension (t)
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, t), # Target shape (Freq, Time)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)
            
            # Resize mask to match new frequency dimension
            # Expand mask to match mel_target shape (B, T, F')
            mask = mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T, F')
            
        elif self.current_stage == 2:
            # Downsample ground truth for mid-res stage
            scale_factor = 1/self.config['model']['mid_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)
            
            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # Correct unpacking
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Upsample time dimension by factor of 2 (assuming 862*2=1724)
            target_time_dim = t * 2
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, target_time_dim), # Target shape (Freq, Time * 2)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T*2)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)
            
            # Upsample mask along time dimension to match target_time_dim (T*2)
            # Original mask shape: (B, T) where T=862
            # Target mask shape: (B, T*2, F') where T*2=1724
            mask_float = mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            upsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time*2)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T*2)

            # Expand upsampled mask to match mel_target frequency dimension (F')
            mask = upsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T*2, F')
            
        else:
            # Full resolution for final stage
            mel_target = mel_specs
            # Expand mask to match mel_target shape (B, T, F)
            mask = mask.unsqueeze(2).expand(-1, -1, mel_specs.shape[2]) # Expand to (B, T, F)
        
        # Handle dimensionality issues
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
            
        if mel_target.dim() == 3 and mel_pred.dim() == 2:
            mel_pred = mel_pred.unsqueeze(0)
        
        # Compute element-wise loss
        loss_elementwise = self.loss_fn(mel_pred, mel_target)

        # --- Apply Weighted Loss ---
        unvoiced_weight = 2.0 # Weight for unvoiced frames
        # Ensure voiced_mask has shape (B, T) before unsqueezing
        if voiced_mask.dim() == 3 and voiced_mask.shape[-1] == 1:
             voiced_mask = voiced_mask.squeeze(-1) # Make it (B, T)
        # Create weights: shape (B, T, 1), broadcastable to (B, T, F)
        loss_weights = torch.where(voiced_mask.unsqueeze(-1), 1.0, unvoiced_weight)
        loss_weights = loss_weights.to(loss_elementwise.device) # Ensure same device

        # Apply length mask and loss weights
        # mask has shape (B, T, F)
        weighted_loss = loss_elementwise * mask.float() * loss_weights

        # Normalize by the sum of applied weights within the mask
        # This keeps the loss magnitude somewhat consistent
        total_weight_sum = (mask.float() * loss_weights).sum()
        loss = weighted_loss.sum() / total_weight_sum.clamp(min=1e-8)
        # --- End Weighted Loss ---
        
        self.log('train_loss', loss, prog_bar=True)
        return loss # Revert to returning raw tensor for diagnostics
    
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
        
        # Create masks for variable length
        max_len = mel_specs.shape[1] # Use time dimension (dim 1)
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        
        # Adjust target resolution based on current stage
        if self.current_stage == 1:
            # Downsample ground truth for low-res stage
            scale_factor = 1/self.config['model']['low_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)
            
            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # Correct unpacking
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Downsample keeping original time dimension (t)
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, t), # Target shape (Freq, Time)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)
            
            # Resize mask to match new frequency dimension
            # Expand mask to match mel_target shape (B, T, F')
            mask = mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T, F')
            
        elif self.current_stage == 2:
            # Downsample ground truth for mid-res stage
            scale_factor = 1/self.config['model']['mid_res_scale']
            freq_dim = int(self.config['model']['mel_bins'] * scale_factor)
            
            # Reshape to [B, 1, F, T] for 2D interpolation (F=Freq, T=Time)
            b, t, c = mel_specs.shape # Correct unpacking
            mel_specs_permuted = mel_specs.permute(0, 2, 1) # Shape (B, F, T)
            mel_specs_4d = mel_specs_permuted.unsqueeze(1) # Shape (B, 1, F, T)

            # Upsample time dimension by factor of 2 (assuming 862*2=1724)
            target_time_dim = t * 2
            mel_target = F.interpolate(
                mel_specs_4d,
                size=(freq_dim, target_time_dim), # Target shape (Freq, Time * 2)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to (B, F', T*2)

            # Permute back to (B, T, F')
            mel_target = mel_target.permute(0, 2, 1)
            
            # Upsample mask along time dimension to match target_time_dim (T*2)
            # Original mask shape: (B, T) where T=862
            # Target mask shape: (B, T*2, F') where T*2=1724
            mask_float = mask.float().unsqueeze(1).unsqueeze(1) # Shape (B, 1, 1, T)
            upsampled_mask = F.interpolate(
                mask_float,
                size=(1, target_time_dim), # Target shape (1, Time*2)
                mode='nearest'
            ).squeeze(1).squeeze(1) # Back to (B, T*2)

            # Expand upsampled mask to match mel_target frequency dimension (F')
            mask = upsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # Expand to (B, T*2, F')
            
        else:
            # Full resolution for final stage
            mel_target = mel_specs
            # Expand mask to match mel_target shape (B, T, F)
            mask = mask.unsqueeze(2).expand(-1, -1, mel_specs.shape[2]) # Expand to (B, T, F)
        
        # Handle dimensionality issues
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1) # Shape becomes [B, F, T_pred]
            # Permute mel_pred to match mel_target shape [B, T, F]
            mel_pred = mel_pred.permute(0, 2, 1) # Shape becomes [B, T_pred, F]
            
        if mel_target.dim() == 3 and mel_pred.dim() == 2:
            # This condition should no longer be met after permutation
            mel_pred = mel_pred.unsqueeze(0)
        
        # --- End logging ---
        
        # Compute element-wise loss
        loss_elementwise = self.loss_fn(mel_pred, mel_target)

        # --- Apply Weighted Loss ---
        unvoiced_weight = 2.0 # Weight for unvoiced frames (same as training)
        # Ensure voiced_mask has shape (B, T) before unsqueezing
        if voiced_mask.dim() == 3 and voiced_mask.shape[-1] == 1:
             voiced_mask = voiced_mask.squeeze(-1) # Make it (B, T)
        # Create weights: shape (B, T, 1), broadcastable to (B, T, F)
        loss_weights = torch.where(voiced_mask.unsqueeze(-1), 1.0, unvoiced_weight)
        loss_weights = loss_weights.to(loss_elementwise.device) # Ensure same device

        # Apply length mask and loss weights
        # mask has shape (B, T, F)
        weighted_loss = loss_elementwise * mask.float() * loss_weights

        # Normalize by the sum of applied weights within the mask
        total_weight_sum = (mask.float() * loss_weights).sum()
        loss = weighted_loss.sum() / total_weight_sum.clamp(min=1e-8)
        # --- End Weighted Loss ---
        self.log('val_loss', loss, prog_bar=True)
        
        # Visualize first sample in batch at appropriate intervals
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            # Get the actual length
            length = lengths[0].item()
            self._log_mel_comparison(
                mel_pred[0, :, :length].detach().cpu(), 
                mel_target[0, :, :length].detach().cpu()
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
        
        # Convert to tensor
        img = Image.open(buf)
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # Log to tensorboard
        self.logger.experiment.add_image(
            f'mel_comparison_stage{self.current_stage}', 
            img_tensor, 
            self.current_epoch
        )
        
        plt.close(fig)
    
    def configure_optimizers(self):
        # Determine the learning rate for the current stage
        stage_key = f'stage{self.current_stage}'
        try:
            current_lr = self.config['train']['learning_rate_per_stage'][stage_key]
            print(f"Using learning rate {current_lr} for stage {self.current_stage}") # Added logging
        except KeyError:
            # Fallback or error handling if the key is missing
            print(f"Warning: Learning rate for '{stage_key}' not found in config. Falling back to default 0.001.")
            # Using a default LR as fallback
            current_lr = 0.001

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=current_lr, # Use the stage-specific learning rate
            weight_decay=self.config['train']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['train']['lr_factor'],
            patience=self.config['train']['lr_patience'],
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }