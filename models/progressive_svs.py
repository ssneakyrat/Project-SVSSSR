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
    # [No changes to this class]
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
        
        # Use original high-res channels from config to maintain checkpoint compatibility
        # DO NOT MODIFY CHANNEL SIZES - this ensures checkpoint compatibility
        high_res_channels = config['model']['high_res_channels']
        
        # Stage 1: Low-Resolution Model (20×216)
        self.low_res_model = LowResModel(
            input_dim, 
            config['model']['low_res_channels'], 
            output_dim=low_res_freq_bins
        )
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        self.mid_res_upsampler = MidResUpsampler(
            low_res_freq_bins,
            config['model']['mid_res_channels'], 
            output_dim=mid_res_freq_bins
        )
        
        # Stage 3: High-Resolution Upsampler (80×864)
        self.high_res_upsampler = HighResUpsampler(
            mid_res_freq_bins,
            high_res_channels, 
            output_dim=full_mel_bins
        )
        
        # Loss function
        self.loss_fn = nn.L1Loss(reduction='none')
        
        # Adjust learning rates for different stages
        self.stage_lr_scale = {
            1: 1.0,    # Full learning rate for Stage 1
            2: 0.5,    # Half learning rate for Stage 2
            3: 0.1     # 1/10th learning rate for Stage 3
        }
        
        # For tracking metrics across batches
        self.last_batch_stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
        
    def forward(self, f0, phone_label, phone_duration, midi_label):
        # Encode input features
        features = self.feature_encoder(f0, phone_label, phone_duration, midi_label)
        
        # Stage 1: Low-Resolution Model (20×216)
        low_res_output = self.low_res_model(features)
        
        if self.current_stage == 1:
            return low_res_output
        
        # Stage 2: Mid-Resolution Upsampler (40×432)
        # Ensure the input shape is correct for mid_res_upsampler
        if low_res_output.dim() != 3:
            print(f"WARNING: Unexpected low_res_output shape: {low_res_output.shape}")
            # Attempt to reshape if needed
            if low_res_output.dim() == 4 and low_res_output.size(1) == 1:
                low_res_output = low_res_output.squeeze(1)
        
        mid_res_output = self.mid_res_upsampler(low_res_output)
        
        if self.current_stage == 2:
            return mid_res_output
        
        # Stage 3: High-Resolution Upsampler (80×864)
        # Ensure mid_res_output is in the right shape for high_res_upsampler
        if mid_res_output.dim() != 4 or mid_res_output.size(1) != 1:
            print(f"WARNING: Unexpected mid_res_output shape: {mid_res_output.shape}")
            # Log more details about the tensor
            if mid_res_output.dim() == 3:
                # Assuming [B, C, T] -> needs to be [B, 1, C, T]
                print("Reshaping mid_res_output from 3D to 4D")
                mid_res_output = mid_res_output.unsqueeze(1)
        
        # Save input for residual connection (implemented in HighResUpsampler)
        high_res_output = self.high_res_upsampler(mid_res_output)
        
        # NEW: Add value boost for stage 3 outputs to make them more visible
        # This scaling helps ensure the output isn't near-zero (invisible)
        if self.current_stage == 3:
            # Apply a mild scaling boost to make output more visible
            high_res_output = high_res_output * 1.5
        
        return high_res_output
    
    def _calculate_stage_specific_loss(self, pred, target, mask=None):
        """
        Calculate stage-specific loss with adaptive weighting
        """
        # Stage 3 may need additional stabilization with regularization
        loss = self.loss_fn(pred, target)
        
        if mask is not None:
            # Apply mask to loss - only consider valid timesteps
            loss = loss * mask.float()
            # Normalize by actual lengths
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        # Add stability regularization for Stage 3, but with reduced weight for better detail
        if self.current_stage == 3:
            # Further reduce L2 regularization to allow more detailed predictions
            reg_loss = 0.000005 * (pred ** 2).mean()  # Reduced from 0.00001
            
            # Add spectral convergence loss component
            if pred.dim() == 3 and target.dim() == 3:
                # Get spectrogram magnitude
                pred_mag = torch.abs(pred)
                target_mag = torch.abs(target)
                
                # Calculate spectral convergence: ||X - Y||_F / ||Y||_F
                # where ||.||_F is the Frobenius norm
                sc_loss = torch.norm(pred_mag - target_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
                
                # Increase spectral convergence weight for better results
                loss = loss + 0.3 * sc_loss  # Increased from 0.2
            
            loss = loss + reg_loss
            
        return loss
    
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
        
        # Enhanced debug info - MORE DETAILED
        if batch_idx == 0:
            print(f"Stage {self.current_stage} - Original mel shape: {mel_specs.shape}")
            print(f"Stage {self.current_stage} - Prediction shape: {mel_pred.shape}")
            
            # Add more detailed debug info for all stages, especially Stage 3
            curr_mean = mel_pred.mean().item()
            curr_std = mel_pred.std().item()
            curr_min = mel_pred.min().item()
            curr_max = mel_pred.max().item()
            
            # NEW: Calculate additional percentile statistics for better understanding
            flat_pred = mel_pred.detach().cpu().flatten().numpy()
            p05 = np.percentile(flat_pred, 5)
            p25 = np.percentile(flat_pred, 25) 
            p50 = np.percentile(flat_pred, 50)
            p75 = np.percentile(flat_pred, 75)
            p95 = np.percentile(flat_pred, 95)
            
            print(f"Stage {self.current_stage} - Prediction stats:")
            print(f"  Mean={curr_mean:.6f}, Std={curr_std:.6f}")
            print(f"  Min={curr_min:.6f}, Max={curr_max:.6f}")
            print(f"  Percentiles: p05={p05:.6f}, p25={p25:.6f}, p50={p50:.6f}, p75={p75:.6f}, p95={p95:.6f}")
            
            # Check if prediction is changing between batches
            if hasattr(self, 'last_batch_stats'):
                prev_mean = self.last_batch_stats['mean']
                prev_std = self.last_batch_stats['std']
                prev_min = self.last_batch_stats['min']
                prev_max = self.last_batch_stats['max']
                
                print(f"Prediction stats change - Mean: {curr_mean-prev_mean:.6f}, Std: {curr_std-prev_std:.6f}")
                print(f"                          Min: {curr_min-prev_min:.6f}, Max: {curr_max-prev_max:.6f}")
            
            self.last_batch_stats = {
                'mean': curr_mean,
                'std': curr_std,
                'min': curr_min,
                'max': curr_max
            }
            
            # NEW: If Stage 3 with near-zero predictions, issue a warning
            if self.current_stage == 3 and curr_max < 0.01:
                print("WARNING: Stage 3 prediction values are very close to zero!")
                print("This may result in blank or nearly invisible visualization output.")
                print("Consider scaling up the output to make it more visible.")
        
        # Handle dimensionality issues (in case prediction is 4D)
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
        
        # FIX: Always resize target to match prediction dimensions, regardless of stage
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
        
        # Debug info
        if batch_idx == 0:
            print(f"Stage {self.current_stage} - Target shape after interpolation: {mel_target.shape}")
            print(f"Stage {self.current_stage} - Mask shape: {mask.shape}")
        
        # Compute loss with stage-specific adaptations
        loss = self._calculate_stage_specific_loss(mel_pred, mel_target, mask)
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Visualize first sample in batch at appropriate intervals
        # For Stage 3, visualize more frequently
        vis_interval = 10 if self.current_stage == 3 else 20  # Increased frequency (5->2 for stage 3)
        if batch_idx == 0 and self.current_epoch % vis_interval == 0:
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
        debug_interval = 5 if self.current_stage == 3 else 20  # More frequent for Stage 3
        if batch_idx % debug_interval == 0:
            print(f"Stage {self.current_stage} - Original mel shape: {mel_specs.shape}")
            print(f"Stage {self.current_stage} - Prediction shape: {mel_pred.shape}")
            
            # Add more detailed debug info for all stages
            print(f"Stage {self.current_stage} - Prediction min/max/mean/std: "
                  f"{mel_pred.min().item():.6f}/{mel_pred.max().item():.6f}/"
                  f"{mel_pred.mean().item():.6f}/{mel_pred.std().item():.6f}")
                
            # For Stage 3, log gradient norms periodically (using on_after_backward instead of direct calls)
            self.log('train_loss_batch', self.trainer.progress_bar_metrics.get('train_loss', 0), prog_bar=False)
        
        # Handle dimensionality issues (in case prediction is 4D)
        if mel_pred.dim() == 4 and mel_pred.size(1) == 1:
            mel_pred = mel_pred.squeeze(1)
        
        # FIX: Always resize target to match prediction dimensions, regardless of stage
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
        
        # Debug occasionally
        if batch_idx % debug_interval == 0:
            print(f"Stage {self.current_stage} - Target shape after interpolation: {mel_target.shape}")
        
        # Compute loss with stage-specific adaptations
        loss = self._calculate_stage_specific_loss(mel_pred, mel_target, mask)
        
        # Log training loss properly (WITHOUT manually calling backward)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def on_after_backward(self):
        """Called after backward pass to log gradients without interfering with PyTorch Lightning"""
        # Only track gradients occasionally and for Stage 3
        if self.current_stage == 3 and self.global_step % 50 == 0:
            grad_norms = {}
            for name, module in [
                ('encoder', self.feature_encoder),
                ('low_res', self.low_res_model),
                ('mid_res', self.mid_res_upsampler),
                ('high_res', self.high_res_upsampler)
            ]:
                total_norm = 0
                param_count = 0
                for p in module.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    total_norm = total_norm ** 0.5
                    grad_norms[f'grad_norm_{name}'] = total_norm
                    print(f"Gradient norm {name}: {total_norm:.6f}")
            
            # Log gradient norms to tensorboard
            self.logger.experiment.add_scalars(
                'gradient_norms',
                grad_norms,
                self.global_step
            )
    
    def _log_mel_comparison(self, pred_mel, target_mel):
        """
        Create and log mel spectrogram comparison visualizations with proper figure cleanup.
        """
        main_fig = None
        
        try:
            # ENHANCED: Improved visualization with dynamic scaling for better visibility
            
            # Get data range for better visualization
            pred_min = pred_mel.min().item()
            pred_max = pred_mel.max().item()
            target_min = target_mel.min().item()
            target_max = target_mel.max().item()
            
            # Set visualization ranges - use dynamic range for Stage 3
            if self.current_stage == 3:
                # Use the actual data range instead of fixed values
                vmin = min(pred_min, target_min)
                # For the maximum, ensure a minimum range even if data is near zero
                actual_vmax = max(pred_max, target_max)
                vmax = max(actual_vmax, 0.1)  # Ensure at least 0.1 range even if data is near zero
                
                # Print visualization ranges to help with debugging
                print(f"Stage 3 visualization ranges - vmin: {vmin:.6f}, vmax: {vmax:.6f}")
                print(f"  (Actual data ranges - pred: [{pred_min:.6f}, {pred_max:.6f}], "
                      f"target: [{target_min:.6f}, {target_max:.6f}])")
            else:
                # For stages 1-2, use standard range
                vmin = 0
                vmax = 1
            
            # Create enhanced visualization for comparison
            main_fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot prediction with dynamic colormap range
            im1 = axes[0].imshow(pred_mel.numpy(), origin='lower', aspect='auto', 
                                vmin=vmin, vmax=vmax, cmap='viridis')
            axes[0].set_title(f'Predicted Mel (Stage {self.current_stage}) - Min: {pred_min:.6f}, Max: {pred_max:.6f}')
            plt.colorbar(im1, ax=axes[0])
            
            # Plot ground truth with same dynamic range
            im2 = axes[1].imshow(target_mel.numpy(), origin='lower', aspect='auto',
                                vmin=vmin, vmax=vmax, cmap='viridis')
            axes[1].set_title(f'Target Mel - Min: {target_min:.6f}, Max: {target_max:.6f}')
            plt.colorbar(im2, ax=axes[1])
            
            # Plot difference with appropriate range
            diff = np.abs(pred_mel.numpy() - target_mel.numpy())
            diff_max = diff.max()
            im3 = axes[2].imshow(diff, origin='lower', aspect='auto', 
                                cmap='hot', vmin=0, vmax=min(diff_max, 0.5))
            axes[2].set_title(f'Absolute Difference - Max: {diff_max:.6f}')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
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
            
            # For Stage 3, add separate frequency band visualizations
            if self.current_stage == 3:
                # Split into frequency bands (lower, middle, upper)
                freq_bands = 3
                band_size = pred_mel.shape[0] // freq_bands
                
                for i in range(freq_bands):
                    band_fig = None
                    try:
                        start_idx = i * band_size
                        end_idx = (i+1) * band_size if i < freq_bands-1 else pred_mel.shape[0]
                        
                        # Get band data for specific min/max visualization
                        pred_band = pred_mel[start_idx:end_idx]
                        target_band = target_mel[start_idx:end_idx]
                        
                        pred_band_min = pred_band.min().item()
                        pred_band_max = pred_band.max().item()
                        target_band_min = target_band.min().item()
                        target_band_max = target_band.max().item()
                        
                        # Use dynamic range for visualization
                        band_vmin = min(pred_band_min, target_band_min)
                        band_vmax = max(max(pred_band_max, target_band_max), 0.1)
                        
                        # Create visualization for this band
                        band_fig, band_axes = plt.subplots(2, 1, figsize=(10, 6))
                        
                        # Plot this frequency band
                        band_axes[0].imshow(pred_band.numpy(), origin='lower', aspect='auto',
                                           vmin=band_vmin, vmax=band_vmax)
                        band_axes[0].set_title(f'Predicted Mel - Band {i+1} (Freq: {start_idx}-{end_idx-1}) '
                                             f'Min: {pred_band_min:.6f}, Max: {pred_band_max:.6f}')
                        
                        band_axes[1].imshow(target_band.numpy(), origin='lower', aspect='auto',
                                           vmin=band_vmin, vmax=band_vmax)
                        band_axes[1].set_title(f'Target Mel - Band {i+1} (Freq: {start_idx}-{end_idx-1}) '
                                             f'Min: {target_band_min:.6f}, Max: {target_band_max:.6f}')
                        
                        plt.tight_layout()
                        
                        # Convert plot to image
                        band_buf = io.BytesIO()
                        plt.savefig(band_buf, format='png')
                        band_buf.seek(0)
                        
                        # Convert to tensor
                        band_img = Image.open(band_buf)
                        band_img_tensor = torchvision.transforms.ToTensor()(band_img)
                        
                        # Log to tensorboard
                        self.logger.experiment.add_image(
                            f'freq_band_{i+1}_stage{self.current_stage}', 
                            band_img_tensor, 
                            self.current_epoch
                        )
                    finally:
                        # Ensure each band figure is closed even if an exception occurs
                        if band_fig is not None:
                            plt.close(band_fig)
        finally:
            # Ensure main figure is closed even if an exception occurs
            if main_fig is not None:
                plt.close(main_fig)
    
    def configure_optimizers(self):
        # Implement stage-specific learning rates
        base_lr = self.config['train']['learning_rate']
        stage_lr = base_lr * self.stage_lr_scale.get(self.current_stage, 1.0)
        
        print(f"Using stage-specific learning rate: {stage_lr} (base: {base_lr}, scale: {self.stage_lr_scale.get(self.current_stage, 1.0)})")
        
        # For Stage 3, use AdamW with slightly different parameters
        if self.current_stage == 3:
            # Higher learning rate specifically for high_res_upsampler parameters
            high_res_params = []
            other_params = []
            
            # Separate parameters for different learning rates
            for name, param in self.named_parameters():
                if 'high_res_upsampler' in name:
                    high_res_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': high_res_params, 'lr': stage_lr * 2.0},  # Double LR for high_res
                {'params': other_params, 'lr': stage_lr}
            ]
            
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config['train']['weight_decay'] * 0.1,  # Reduce weight decay
                eps=1e-5,  # Increase epsilon for better stability
                betas=(0.9, 0.999)  # Default betas, but being explicit
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=stage_lr,
                weight_decay=self.config['train']['weight_decay']
            )
        
        # Increase scheduler patience for Stage 3
        patience = self.config['train']['lr_patience']
        if self.current_stage == 3:
            patience = max(patience, 10)  # More patience for Stage 3
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['train']['lr_factor'],
            patience=patience,
            verbose=True,
            min_lr=1e-6  # Add a minimum lr to prevent going too low
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
        
    def on_train_start(self):
        # Optionally freeze earlier stages when training Stage 3
        if self.current_stage == 3:
            # Check if we should freeze earlier stages (could be a config option)
            freeze_earlier_stages = self.config['train'].get('freeze_earlier_stages', False)
            
            if freeze_earlier_stages:
                print("Freezing weights for Stage 1 and Stage 2 components")
                # Freeze feature encoder (partial)
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
                
                # Freeze low-res model
                for param in self.low_res_model.parameters():
                    param.requires_grad = False
                
                # Freeze mid-res upsampler
                for param in self.mid_res_upsampler.parameters():
                    param.requires_grad = False