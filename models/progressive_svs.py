import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision
import numpy as np

from models.blocks import LowResModel, MidResUpsampler, HighResUpsampler

class FeatureEncoder(nn.Module):
    """Encodes F0, phone labels, and MIDI labels into a combined feature vector"""
    def __init__(self, config):
        super().__init__()
        
        self.f0_embed_dim = config['model']['f0_embed_dim']
        self.phone_embed_dim = config['model']['phone_embed_dim']
        self.midi_embed_dim = config['model']['midi_embed_dim']
        
        # F0 encoding (continuous value)
        self.f0_projection = nn.Linear(1, self.f0_embed_dim)
        
        # Phone encoding (categorical)
        num_phones = 100  # Default if not specified
        self.phone_embedding = nn.Embedding(num_phones, self.phone_embed_dim)
        
        # MIDI encoding (categorical)
        num_midi_notes = 128  # MIDI standard
        self.midi_embedding = nn.Embedding(num_midi_notes, self.midi_embed_dim)
        
    def forward(self, f0, phone_label, phone_duration, midi_label):
        batch_size, seq_len = f0.size()
        
        # Normalize F0 between 0-1 for better training
        f0 = (f0 - 50.0) / (600.0 - 50.0)  # Use f0_min and f0_max
        f0 = torch.clamp(f0, 0.0, 1.0)
        
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
    """Progressive Growing Synthesis model for SVS
    
    The model has three stages:
    1. Low-Resolution Model (20×216)
    2. Mid-Resolution Upsampler (40×432)
    3. High-Resolution Upsampler (80×864)
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.current_stage = config['model'].get('current_stage', 1)
        
        # Set explicit dimensions for each stage
        self.full_time_frames = config['model']['time_frames']  # 864
        self.full_mel_bins = config['model']['mel_bins']  # 80
        
        # Low-res dimensions (÷4): 20×216
        self.low_res_scale = config['model']['low_res_scale']  # 4
        self.low_res_freq_bins = self.full_mel_bins // self.low_res_scale
        self.low_res_time_frames = self.full_time_frames // self.low_res_scale
        
        # Mid-res dimensions (÷2): 40×432
        self.mid_res_scale = config['model']['mid_res_scale']  # 2
        self.mid_res_freq_bins = self.full_mel_bins // self.mid_res_scale
        self.mid_res_time_frames = self.full_time_frames // self.mid_res_scale
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(config)
        
        # Calculate total embedding dimension
        input_dim = self.feature_encoder.f0_embed_dim + self.feature_encoder.phone_embed_dim + self.feature_encoder.midi_embed_dim
        
        # Stage 1: Low-Resolution Model (20×216)
        self.low_res_model = LowResModel(
            input_dim,
            config['model']['low_res_channels'],
            output_dim=self.low_res_freq_bins,
            output_time_frames=self.low_res_time_frames # Pass the target time frames
        )
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        self.mid_res_upsampler = MidResUpsampler(
            self.low_res_freq_bins,
            config['model']['mid_res_channels'], 
            output_dim=self.mid_res_freq_bins
        )
        
        # Stage 3: High-Resolution Upsampler (80×864)
        self.high_res_upsampler = HighResUpsampler(
            self.mid_res_freq_bins,
            config['model']['high_res_channels'], 
            output_dim=self.full_mel_bins
        )
        
        # Loss function
        self.loss_fn = nn.L1Loss(reduction='none')
        
    def _apply_length_mask(self, mel_output, lengths):
        """Apply length mask to generated output"""
        batch_size = mel_output.size(0)
        
        # Determine dimensions based on output shape and stage
        if mel_output.dim() == 4:  # [B, 1, F, T]
            freq_dim = mel_output.size(2)
            time_dim = mel_output.size(3)
            # Create time dimension mask
            mask = torch.arange(time_dim, device=lengths.device).expand(batch_size, time_dim) < lengths.unsqueeze(1)
            # Reshape for broadcasting [B, 1, 1, T]
            mask = mask.unsqueeze(1).unsqueeze(1)
            # Expand across frequency dimension [B, 1, F, T]
            mask = mask.expand(-1, -1, freq_dim, -1)
        elif mel_output.dim() == 3:  # [B, F, T]
            freq_dim = mel_output.size(1)
            time_dim = mel_output.size(2)
            # Create time dimension mask
            mask = torch.arange(time_dim, device=lengths.device).expand(batch_size, time_dim) < lengths.unsqueeze(1)
            # Reshape for broadcasting [B, 1, T]
            mask = mask.unsqueeze(1)
            # Expand across frequency dimension [B, F, T]
            mask = mask.expand(-1, freq_dim, -1)
        else:
            return mel_output
        
        # Apply mask: keep values where mask is True, set to 0 otherwise
        masked_output = torch.where(mask, mel_output, torch.zeros_like(mel_output))
        return masked_output

    def forward(self, f0, phone_label, phone_duration, midi_label, lengths=None):
        """Forward pass through the model based on current stage"""
        # Encode input features
        features = self.feature_encoder(f0, phone_label, phone_duration, midi_label)
        
        # Stage 1: Low-Resolution Model (20×216)
        low_res_output = self.low_res_model(features)
        
        if self.current_stage == 1:
            if lengths is not None:
                # Scale lengths for low res
                scaled_lengths = torch.ceil(lengths / self.low_res_scale).long()
                low_res_output = self._apply_length_mask(low_res_output, scaled_lengths)
            return low_res_output
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        # Ensure low_res_output is 3D [B, C, T]
        if low_res_output.dim() == 4 and low_res_output.size(1) == 1:
            low_res_output = low_res_output.squeeze(1)
            
        mid_res_output = self.mid_res_upsampler(low_res_output)
        
        if self.current_stage == 2:
            if lengths is not None:
                # Scale lengths for mid res
                scaled_lengths = torch.ceil(lengths / self.mid_res_scale).long()
                mid_res_output = self._apply_length_mask(mid_res_output, scaled_lengths)
            return mid_res_output
        
        # Stage 3: High-Resolution Upsampler (80×864)
        # Ensure mid_res_output is 4D [B, 1, C, T]
        if mid_res_output.dim() == 3:
            mid_res_output = mid_res_output.unsqueeze(1)
            
        high_res_output = self.high_res_upsampler(mid_res_output)
        
        if lengths is not None:
            high_res_output = self._apply_length_mask(high_res_output, lengths)
        
        return high_res_output
    
    def _get_stage_target(self, mel_specs, stage):
        """Create properly downsampled targets for the current stage"""
        if stage == 1:
            # For Stage 1: 20×216 (downsample by 4x)
            target = F.avg_pool2d(
                mel_specs.unsqueeze(1),
                kernel_size=(self.low_res_scale, self.low_res_scale),
                stride=(self.low_res_scale, self.low_res_scale)
            ).squeeze(1)
        elif stage == 2:
            # For Stage 2: 40×432 (downsample by 2x)
            target = F.avg_pool2d(
                mel_specs.unsqueeze(1),
                kernel_size=(self.mid_res_scale, self.mid_res_scale),
                stride=(self.mid_res_scale, self.mid_res_scale)
            ).squeeze(1)
        else:
            # Stage 3: Full resolution 80×864
            target = mel_specs
        
        return target
    
    def _resize_mask(self, mask, target_length):
        """Resize mask to match target sequence length"""
        if mask.shape[-1] == target_length:
            return mask
            
        return F.interpolate(
            mask.float().unsqueeze(1),
            size=target_length,
            mode='nearest'
        ).squeeze(1).bool()
    
    def training_step(self, batch, batch_idx):
        """Training step for one batch"""
        mel_specs = batch['mel_spec']  # [B, 80, 864]
        f0 = batch['f0']  # [B, 864]
        phone_label = batch['phone_label']  # [B, 864]
        phone_duration = batch['phone_duration']  # List of durations
        midi_label = batch['midi_label']  # [B, 864]
        lengths = batch['length']  # [B]
        
        # Forward pass
        mel_pred = self(f0, phone_label, phone_duration, midi_label, lengths)
        
        # Get proper target for current stage
        mel_target = self._get_stage_target(mel_specs, self.current_stage)
        
        # Calculate appropriate sequence lengths for current stage
        if self.current_stage == 1:
            target_lengths = torch.ceil(lengths / self.low_res_scale).long()
            target_seq_len = self.low_res_time_frames
        elif self.current_stage == 2:
            target_lengths = torch.ceil(lengths / self.mid_res_scale).long()
            target_seq_len = self.mid_res_time_frames
        else:
            target_lengths = lengths
            target_seq_len = self.full_time_frames
        
        # Create mask for variable length sequences
        mask = torch.arange(target_seq_len, device=lengths.device).expand(len(target_lengths), target_seq_len) < target_lengths.unsqueeze(1)
        
        # Ensure dimensions match between prediction and target
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
        
        # Apply mask based on dimensions
        # Note: The specific interpolation for stage 1 prediction is removed
        # as it's now handled by the generic resizing below if needed.
        # The mask expansion is also moved below after resizing.

        # Ensure pred and target have the same length (target_seq_len) before loss
        current_len_pred = mel_pred.shape[-1]
        current_len_target = mel_target.shape[-1]
        
        if current_len_pred != target_seq_len:
            mel_pred = F.interpolate(mel_pred, size=target_seq_len, mode='linear', align_corners=False)
            
        if current_len_target != target_seq_len:
             mel_target = F.interpolate(mel_target, size=target_seq_len, mode='linear', align_corners=False)

        # Resize mask if needed after target interpolation
        if mask.shape[-1] != target_seq_len:
             mask = self._resize_mask(mask, target_seq_len) # Use existing resize function - mask is still [B, T]

        # Calculate loss
        loss = self.loss_fn(mel_pred, mel_target) # Shape [B, C, T]

        # Expand mask to match loss shape [B, C, T] before multiplication
        if loss.dim() == 3 and mask.dim() == 2:
             mask = mask.unsqueeze(1).expand_as(loss) # Expand to [B, C, T]

        # Apply mask
        loss = loss * mask.float()
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch"""
        mel_specs = batch['mel_spec']
        f0 = batch['f0']
        phone_label = batch['phone_label']
        phone_duration = batch['phone_duration']
        midi_label = batch['midi_label']
        lengths = batch['length']
        
        # Forward pass
        mel_pred = self(f0, phone_label, phone_duration, midi_label, lengths)
        
        # Get proper target for current stage
        mel_target = self._get_stage_target(mel_specs, self.current_stage)
        
        # Calculate appropriate sequence lengths for current stage
        if self.current_stage == 1:
            target_lengths = torch.ceil(lengths / self.low_res_scale).long()
            target_seq_len = self.low_res_time_frames
        elif self.current_stage == 2:
            target_lengths = torch.ceil(lengths / self.mid_res_scale).long()
            target_seq_len = self.mid_res_time_frames
        else:
            target_lengths = lengths
            target_seq_len = self.full_time_frames
        
        # Create mask for variable length sequences
        mask = torch.arange(target_seq_len, device=lengths.device).expand(len(target_lengths), target_seq_len) < target_lengths.unsqueeze(1)
        
        # Ensure dimensions match between prediction and target
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
            
        # Apply mask based on dimensions
        # Note: The specific interpolation for stage 1 prediction is removed
        # as it's now handled by the generic resizing below if needed.
        # The mask expansion is also moved below after resizing.
            
        # Ensure pred and target have the same length (target_seq_len) before loss
        current_len_pred = mel_pred.shape[-1]
        current_len_target = mel_target.shape[-1]
        
        if current_len_pred != target_seq_len:
            mel_pred = F.interpolate(mel_pred, size=target_seq_len, mode='linear', align_corners=False)
            
        if current_len_target != target_seq_len:
             mel_target = F.interpolate(mel_target, size=target_seq_len, mode='linear', align_corners=False)

        # Resize mask if needed after target interpolation
        if mask.shape[-1] != target_seq_len:
             mask = self._resize_mask(mask, target_seq_len) # Use existing resize function - mask is still [B, T]

        # Calculate loss
        loss = self.loss_fn(mel_pred, mel_target) # Shape [B, C, T]

        # Expand mask to match loss shape [B, C, T] before multiplication
        if loss.dim() == 3 and mask.dim() == 2:
             mask = mask.unsqueeze(1).expand_as(loss) # Expand to [B, C, T]

        # Apply mask
        loss = loss * mask.float()
        loss = loss.sum() / (mask.sum() + 1e-8)
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Visualize first sample in batch if this is the first batch in the epoch
        if batch_idx == 0:
            self._log_mel_comparison(
                mel_pred[0].detach().cpu(), 
                mel_target[0].detach().cpu()
            )
        
        return loss
    
    def _log_mel_comparison(self, pred_mel, target_mel):
        """Create and log mel spectrogram comparison visualizations"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Handle different tensor shapes
        if pred_mel.dim() == 3 and pred_mel.size(0) == 1:
            pred_mel = pred_mel.squeeze(0)
            
        # Plot prediction
        axes[0].imshow(pred_mel.numpy(), origin='lower', aspect='auto')
        axes[0].set_title(f'Predicted Mel (Stage {self.current_stage})')
        
        # Plot ground truth
        axes[1].imshow(target_mel.numpy(), origin='lower', aspect='auto')
        axes[1].set_title('Target Mel')
        
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
        """Configure optimizers with stage-specific learning rates"""
        # Adjust learning rate based on stage
        base_lr = self.config['train']['learning_rate']
        if self.current_stage == 1:
            lr = base_lr
        elif self.current_stage == 2:
            lr = base_lr * 0.5  # Half the learning rate
        else:
            lr = base_lr * 0.1  # 1/10th the learning rate
        
        # Handle freezing earlier stages
        if self.current_stage > 1 and self.config['train'].get('freeze_earlier_stages', False):
            # Freeze earlier stages
            params = []
            if self.current_stage == 2:
                # Only train mid-res upsampler
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
                for param in self.low_res_model.parameters():
                    param.requires_grad = False
                params.extend(list(self.mid_res_upsampler.parameters()))
            else:  # stage 3
                # Only train high-res upsampler
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
                for param in self.low_res_model.parameters():
                    param.requires_grad = False
                for param in self.mid_res_upsampler.parameters():
                    param.requires_grad = False
                params.extend(list(self.high_res_upsampler.parameters()))
        else:
            # Train all parameters
            params = self.parameters()
        
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
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