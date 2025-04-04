import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

from models.blocks import LowResModel, MidResUpsampler, HighResUpsampler

class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.f0_embed_dim = config['model']['f0_embed_dim']
        self.phone_embed_dim = config['model']['phone_embed_dim']
        self.midi_embed_dim = config['model']['midi_embed_dim']
        
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
        
    def forward(self, f0, phone_label, phone_duration, midi_label):
        batch_size, seq_len = f0.size()
        
        # Encode F0 (frame-level)
        f0 = f0.unsqueeze(-1)  # Add feature dimension [B, T, 1]
        f0_encoded = self.f0_projection(f0)  # [B, T, F0_dim]
        
        # Encode phone (frame-level)
        phone_encoded = self.phone_embedding(phone_label)  # [B, T, P_dim]
        
        # Encode MIDI (frame-level)
        midi_encoded = self.midi_embedding(midi_label)  # [B, T, M_dim]
        
        # Concatenate all features [B, T, F0_dim + P_dim + M_dim]
        features = torch.cat([f0_encoded, phone_encoded, midi_encoded], dim=2)
        
        # Reshape to [B, F, T] for 1D convolution
        features = features.transpose(1, 2)
        
        return features

class ProgressiveSVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.current_stage = config['model'].get('current_stage', 1)
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(config)
        
        # Progressive stages
        input_dim = self.feature_encoder.f0_embed_dim + self.feature_encoder.phone_embed_dim + self.feature_encoder.midi_embed_dim
        
        # Calculate frequency dimensions for each stage
        full_mel_bins = config['model']['mel_bins']
        low_res_freq_bins = int(full_mel_bins / config['model']['low_res_scale'])
        mid_res_freq_bins = int(full_mel_bins / config['model']['mid_res_scale'])
        
        # Stage 1: Low-Resolution Model (20×216)
        self.low_res_model = LowResModel(
            input_dim, 
            config['model']['low_res_channels'], 
            output_dim=low_res_freq_bins  # Now using correct dimension
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
        
    def forward(self, f0, phone_label, phone_duration, midi_label):
        # Encode input features
        features = self.feature_encoder(f0, phone_label, phone_duration, midi_label)
        
        # Stage 1: Low-Resolution Model (20×216)
        low_res_output = self.low_res_model(features)
        
        if self.current_stage == 1:
            return low_res_output
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        mid_res_output = self.mid_res_upsampler(low_res_output)
        
        if self.current_stage == 2:
            return mid_res_output
        
        # Stage 3: High-Resolution Upsampler (80×864)
        high_res_output = self.high_res_upsampler(mid_res_output)
        
        return high_res_output
    
    def validation_step(self, batch, batch_idx):
        mel_specs = batch['mel_spec']
        f0 = batch['f0']
        phone_label = batch['phone_label']
        phone_duration = batch['phone_duration']
        midi_label = batch['midi_label']
        lengths = batch['length']
        
        # Forward pass
        mel_pred = self(f0, phone_label, phone_duration, midi_label)
        
        # Create masks for variable length
        max_len = mel_specs.shape[2]
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        
        # Debug info
        if batch_idx == 0:
            print(f"Stage {self.current_stage} - Original mel shape: {mel_specs.shape}")
            print(f"Stage {self.current_stage} - Prediction shape: {mel_pred.shape}")
        
        # Handle dimensionality issues (in case prediction is 4D)
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
        
        # Adjust target resolution based on current stage
        if self.current_stage <= 2:  # For both stage 1 and 2
            # Get target dimensions directly from the prediction
            target_freq_dim = mel_pred.shape[1]
            target_time_dim = mel_pred.shape[2]
            
            # Debug info
            if batch_idx == 0:
                print(f"Stage {self.current_stage} - Target dimensions: freq={target_freq_dim}, time={target_time_dim}")
            
            # Reshape to [B, 1, C, T] for 2D interpolation
            b, c, t = mel_specs.shape
            mel_specs_reshaped = mel_specs.unsqueeze(1)  # [B, 1, C, T]
            
            # Use explicit size for interpolation to match prediction dimensions
            mel_target = torch.nn.functional.interpolate(
                mel_specs_reshaped,
                size=(target_freq_dim, target_time_dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Squeeze back to [B, C', T']
            
            # Also resize mask to match the new time dimension
            new_mask = torch.nn.functional.interpolate(
                mask.float().unsqueeze(1),
                size=(target_time_dim),
                mode='nearest'
            ).squeeze(1).bool()
            mask = new_mask.unsqueeze(1).expand(-1, target_freq_dim, -1)
            
        else:
            # Full resolution for final stage
            mel_target = mel_specs
            mask = mask.unsqueeze(1).expand(-1, mel_specs.shape[1], -1)
        
        # Debug info
        if batch_idx == 0:
            print(f"Stage {self.current_stage} - Target shape after interpolation: {mel_target.shape}")
            print(f"Stage {self.current_stage} - Mask shape: {mask.shape}")
        
        # Compute masked loss
        loss = self.loss_fn(mel_pred, mel_target)
        
        # Apply mask to loss - only consider valid timesteps
        loss = loss * mask.float()
        
        # Normalize by actual lengths
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Visualize first sample in batch at appropriate intervals
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            # Use shape from prediction for visualization
            self._log_mel_comparison(
                mel_pred[0].detach().cpu(), 
                mel_target[0].detach().cpu()
            )
        
        return loss

    def training_step(self, batch, batch_idx):
        mel_specs = batch['mel_spec']
        f0 = batch['f0']
        phone_label = batch['phone_label']
        phone_duration = batch['phone_duration']
        midi_label = batch['midi_label']
        lengths = batch['length']
        
        # Forward pass
        mel_pred = self(f0, phone_label, phone_duration, midi_label)
        
        # Create masks for variable length
        max_len = mel_specs.shape[2]
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        
        # Print debug info occasionally
        if batch_idx % 50 == 0:
            print(f"Stage {self.current_stage} - Original mel shape: {mel_specs.shape}")
            print(f"Stage {self.current_stage} - Prediction shape: {mel_pred.shape}")
        
        # Handle dimensionality issues (in case prediction is 4D)
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
        
        # Adjust target resolution based on current stage
        if self.current_stage <= 2:  # For both stage 1 and 2
            # Get target dimensions directly from the prediction
            target_freq_dim = mel_pred.shape[1]
            target_time_dim = mel_pred.shape[2]
            
            # Reshape to [B, 1, C, T] for 2D interpolation
            b, c, t = mel_specs.shape
            mel_specs_reshaped = mel_specs.unsqueeze(1)  # [B, 1, C, T]
            
            # Use explicit size for interpolation to match prediction dimensions
            mel_target = torch.nn.functional.interpolate(
                mel_specs_reshaped,
                size=(target_freq_dim, target_time_dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Squeeze back to [B, C', T']
            
            # Also resize mask to match the new time dimension
            new_mask = torch.nn.functional.interpolate(
                mask.float().unsqueeze(1),
                size=(target_time_dim),
                mode='nearest'
            ).squeeze(1).bool()
            mask = new_mask.unsqueeze(1).expand(-1, target_freq_dim, -1)
            
        else:
            # Full resolution for final stage
            mel_target = mel_specs
            mask = mask.unsqueeze(1).expand(-1, mel_specs.shape[1], -1)
        
        # Debug occasionally
        if batch_idx % 50 == 0:
            print(f"Stage {self.current_stage} - Target shape after interpolation: {mel_target.shape}")
        
        # Compute masked loss
        loss = self.loss_fn(mel_pred, mel_target)
        
        # Apply mask to loss - only consider valid timesteps
        loss = loss * mask.float()
        
        # Normalize by actual lengths
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def _log_mel_comparison(self, pred_mel, target_mel):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot prediction
        im1 = ax1.imshow(pred_mel.numpy(), origin='lower', aspect='auto')
        ax1.set_title(f'Predicted Mel (Stage {self.current_stage})')
        plt.colorbar(im1, ax=ax1)
        
        # Plot ground truth
        im2 = ax2.imshow(target_mel.numpy(), origin='lower', aspect='auto')
        ax2.set_title('Target Mel')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
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
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config['train']['learning_rate'],
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