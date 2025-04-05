import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.input_processors import PhoneEncoder, F0Encoder, DurationEncoder, MidiEncoder
from models.dilated_backbone import MultiPathDilatedBackbone
from models.feature_mixer import FeatureMixingModule
from models.decoder import MelDecoderModule

class AttentionFreeSVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Extract configuration
        self.mel_bins = config['model']['mel_bins']
        self.variable_length = config['model'].get('variable_length_mode', False)
        
        # Calculate total upsampling factor from configuration
        self.total_upsampling = 1
        for factor in config['model']['upsampling_factors']:
            self.total_upsampling *= factor
        
        print(f"Total upsampling factor: {self.total_upsampling}x")
        
        # Initialize input processing modules
        self.phone_encoder = PhoneEncoder(
            vocab_size=config.get('data', {}).get('phone_vocab_size', 50),
            embed_dim=config['model']['phone_embed_dim'],
            dropout=config['model']['dropout_rate']
        )
        
        self.f0_encoder = F0Encoder(
            input_dim=1,
            hidden_dim=config['model']['f0_embed_dim']
        )
        
        self.duration_encoder = DurationEncoder(
            input_dim=1,
            hidden_dim=config['model']['duration_embed_dim'],
            dropout=config['model']['dropout_rate']
        )
        
        self.midi_encoder = MidiEncoder(
            vocab_size=config.get('data', {}).get('midi_vocab_size', 128),
            embed_dim=config['model']['midi_embed_dim'],
            dropout=config['model']['dropout_rate']
        )
        
        # Calculate total input dimension
        total_input_dim = (
            config['model']['phone_embed_dim'] + 
            config['model']['f0_embed_dim'] + 
            config['model']['duration_embed_dim'] + 
            config['model']['midi_embed_dim']
        )
        
        # Initialize backbone
        self.backbone = MultiPathDilatedBackbone(
            input_dim=total_input_dim,
            num_paths=config['model']['num_paths'],
            path1_config={
                'kernel_sizes': config['model']['path1_kernel_sizes'],
                'dilation_rates': config['model']['path1_dilation_rates'],
                'channels': config['model']['path1_channels']
            },
            path2_config={
                'kernel_sizes': config['model']['path2_kernel_sizes'],
                'dilation_rates': config['model']['path2_dilation_rates'],
                'channels': config['model']['path2_channels']
            },
            path3_config={
                'kernel_sizes': config['model']['path3_kernel_sizes'],
                'dilation_rates': config['model']['path3_dilation_rates'],
                'channels': config['model']['path3_channels']
            }
        )
        
        # Calculate backbone output dimension
        backbone_output_dim = (
            config['model']['path1_channels'] + 
            config['model']['path2_channels'] + 
            config['model']['path3_channels']
        )
        
        # Initialize feature mixing module
        self.feature_mixer = FeatureMixingModule(
            channels=backbone_output_dim,
            kernel_size=config['model']['feature_mixing_kernel'],
            use_depth_wise=config['model']['use_depth_wise']
        )
        
        # Initialize decoder
        self.decoder = MelDecoderModule(
            input_dim=backbone_output_dim,
            mel_bins=self.mel_bins,
            upsampling_factors=config['model']['upsampling_factors']
        )
    
    def forward(self, phone_ids, f0, durations, midi_ids):
        """
        Forward pass through the model
        
        Args:
            phone_ids: Phone identifiers [batch, time]
            f0: Fundamental frequency contour [batch, time, 1]
            durations: Phone durations [batch, time, 1]
            midi_ids: MIDI note identifiers [batch, time]
            
        Returns:
            Predicted mel spectrogram [batch, mel_bins, time*upsampling_factor]
        """
        # Process inputs
        phone_features = self.phone_encoder(phone_ids)
        f0_features = self.f0_encoder(f0)
        duration_features = self.duration_encoder(durations)
        midi_features = self.midi_encoder(midi_ids)
        
        # Concatenate features
        combined_features = torch.cat(
            [phone_features, f0_features, duration_features, midi_features], 
            dim=-1
        )
        
        # Pass through backbone
        backbone_features = self.backbone(combined_features)
        
        # Mix features
        mixed_features = self.feature_mixer(backbone_features)
        
        # Generate mel spectrogram
        mel_output = self.decoder(mixed_features)
        
        return mel_output
    
    def adjust_mel_length(self, mel, target_length=None, downsample=True):
        """
        Adjust the time dimension of mel spectrogram either by downsampling
        to match target length or by a fixed factor.
        
        Args:
            mel: Mel spectrogram tensor of any shape where the last dimension is time
                 and second-to-last is mel_bins (if it exists)
            target_length: Target length for time dimension, if None use total_upsampling
            downsample: If True, downsample; if False, truncate
            
        Returns:
            Adjusted mel spectrogram with same dimensions but modified time length
        """
        # Store original shape for debugging
        original_shape = mel.shape
        original_dim = mel.dim()
        
        # Convert to 4D [batch, channels, mel_bins, time] for adaptive_avg_pool2d
        if original_dim == 1:  # [time] - very unlikely but handle it
            mel = mel.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, time]
        elif original_dim == 2:  # [mel_bins, time] or [time, mel_bins]
            # Check if mel_bins should be the first or second dimension based on shape
            if mel.shape[0] == self.mel_bins:  # [mel_bins, time]
                mel = mel.unsqueeze(0).unsqueeze(0)  # [1, 1, mel_bins, time]
            else:  # [time, mel_bins] or other 2D shape
                # Assume first dim is time if it doesn't match mel_bins
                mel = mel.transpose(0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, mel_bins, time]
        elif original_dim == 3:  # [batch, mel_bins, time]
            mel = mel.unsqueeze(1)  # [batch, 1, mel_bins, time]
        
        # If no target_length provided, downsample by total_upsampling factor
        if target_length is None and downsample:
            # Calculate target length based on upsampling factor
            target_length = mel.shape[-1] // self.total_upsampling
            # Ensure target_length is at least 1
            target_length = max(1, target_length)
        
        # If target_length is specified, adjust to that length
        if target_length is not None:
            if downsample:
                # Use adaptive pooling to downsample - ensure we have 4D tensor for pooling
                try:
                    mel = F.adaptive_avg_pool2d(mel, (mel.shape[-2], target_length))
                except Exception as e:
                    print(f"Error during adaptive_avg_pool2d: {e}")
                    print(f"Original shape: {original_shape}, Current shape: {mel.shape}")
                    print(f"Target length: {target_length}")
                    # Fallback: use simple truncation or padding
                    if mel.shape[-1] > target_length:
                        mel = mel[..., :target_length]
                    elif mel.shape[-1] < target_length:
                        # Pad with zeros
                        padding = torch.zeros(mel.shape[:-1] + (target_length - mel.shape[-1],), 
                                             device=mel.device, dtype=mel.dtype)
                        mel = torch.cat([mel, padding], dim=-1)
            else:
                # Simply truncate to desired length
                if mel.shape[-1] > target_length:
                    mel = mel[..., :target_length]
                elif mel.shape[-1] < target_length:
                    # Pad with zeros
                    padding = torch.zeros(mel.shape[:-1] + (target_length - mel.shape[-1],), 
                                         device=mel.device, dtype=mel.dtype)
                    mel = torch.cat([mel, padding], dim=-1)
        
        # Convert back to original number of dimensions
        if original_dim == 1:
            mel = mel.squeeze(0).squeeze(0).squeeze(0)  # Back to [time]
        elif original_dim == 2:
            if original_shape[0] == self.mel_bins:  # Original was [mel_bins, time]
                mel = mel.squeeze(0).squeeze(0)  # Back to [mel_bins, time]
            else:  # Original was [time, mel_bins]
                mel = mel.squeeze(0).squeeze(0).transpose(0, 1)  # Back to [time, mel_bins]
        elif original_dim == 3:
            mel = mel.squeeze(1)  # Back to [batch, mel_bins, time]
        
        return mel
    
    def compute_masked_loss(self, pred, target, mask=None):
        """
        Compute MSE loss with optional masking for variable-length data
        
        Args:
            pred: Predicted mel spectrogram [batch, mel_bins, time] or [batch, channels, mel_bins, time]
            target: Target mel spectrogram [batch, mel_bins, time] or [batch, channels, mel_bins, time]
            mask: Binary mask for valid time steps [batch, time]
            
        Returns:
            Loss value
        """
        # Handle different tensor shapes
        # If tensors are 3D [batch, mel_bins, time], convert to 4D by adding channel dim
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [batch, mel_bins, time] -> [batch, 1, mel_bins, time]
        
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [batch, mel_bins, time] -> [batch, 1, mel_bins, time]
        
        # Now both should be 4D [batch, channels, mel_bins, time]
        
        # Adjust time dimension mismatch (handle upsampling factor difference)
        time_dim = -1  # Last dimension is time
        if pred.shape[time_dim] != target.shape[time_dim]:
            # Use the general-purpose adjustment method
            pred = self.adjust_mel_length(pred, target.shape[time_dim])
        
        if mask is None:
            return F.mse_loss(pred, target)
        
        # Expand mask to match spectrogram dimensions
        # [batch, time] -> [batch, 1, 1, time]
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Count valid time steps for averaging
        valid_steps = mask.sum()
        if valid_steps == 0:
            valid_steps = 1  # Avoid division by zero
        
        # Compute masked loss
        squared_error = (pred - target) ** 2
        masked_error = squared_error * expanded_mask.float()
        masked_loss = masked_error.sum() / (valid_steps * self.mel_bins)
        
        return masked_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['train']['learning_rate'],
            weight_decay=self.config['train']['weight_decay']
        )
        
        if self.config['train']['lr_scheduler'] == 'reduce_on_plateau':
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
                    'interval': 'epoch'
                }
            }
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """
        Execute a single training step
        
        Args:
            batch: Tuple of (phone_ids, f0, durations, midi_ids, mel_ground_truth[, seq_mask, mel_mask])
            batch_idx: Index of the batch
            
        Returns:
            Training loss
        """
        # Unpack the batch depending on whether using variable length
        if self.variable_length:
            phone_ids, f0, durations, midi_ids, mel_ground_truth, seq_mask, mel_mask = batch
        else:
            phone_ids, f0, durations, midi_ids, mel_ground_truth = batch
            seq_mask, mel_mask = None, None
        
        # Forward pass
        mel_pred = self(phone_ids, f0, durations, midi_ids)
        
        # Calculate loss
        loss = self.compute_masked_loss(mel_pred, mel_ground_truth, mel_mask)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Execute a single validation step
        
        Args:
            batch: Tuple of (phone_ids, f0, durations, midi_ids, mel_ground_truth[, seq_mask, mel_mask])
            batch_idx: Index of the batch
            
        Returns:
            Validation loss
        """
        # Unpack the batch depending on whether using variable length
        if self.variable_length:
            phone_ids, f0, durations, midi_ids, mel_ground_truth, seq_mask, mel_mask = batch
        else:
            phone_ids, f0, durations, midi_ids, mel_ground_truth = batch
            seq_mask, mel_mask = None, None
        
        # Forward pass
        mel_pred = self(phone_ids, f0, durations, midi_ids)
        
        # Calculate loss
        loss = self.compute_masked_loss(mel_pred, mel_ground_truth, mel_mask)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        # Log sample spectrograms at regular intervals
        if batch_idx == 0 and self.global_step > 0:
            # Log first sample in batch
            # Adjust predicted mel to match target length for visualization
            adjusted_mel_pred = self.adjust_mel_length(mel_pred[0], mel_ground_truth[0].shape[-1])
            self._log_spectrograms(adjusted_mel_pred, mel_ground_truth[0])
        
        return loss
    
    def _log_spectrograms(self, pred_mel, target_mel):
        """
        Log spectrograms to TensorBoard with robust handling of different tensor shapes
        
        Args:
            pred_mel: Predicted mel spectrogram
            target_mel: Target mel spectrogram
        """
        import torch
        
        try:
            # Clone tensors to avoid modifying originals
            pred_img = pred_mel.detach().clone()
            target_img = target_mel.detach().clone()
            
            # Handle the case where tensor is [1, time] (flattened)
            if pred_img.dim() == 2 and pred_img.shape[0] == 1:
                # Reshape to [1, mel_bins, time]
                time_frames = pred_img.shape[1] // self.mel_bins
                pred_img = pred_img.reshape(1, self.mel_bins, time_frames)
            
            # Convert to proper format for TensorBoard (CHW)
            if pred_img.dim() == 2:  # [mel_bins, time]
                pred_img = pred_img.unsqueeze(0)  # [1, mel_bins, time]
            elif pred_img.dim() == 3 and pred_img.shape[0] > 1:  # [batch, mel_bins, time]
                pred_img = pred_img[0].unsqueeze(0)  # Take first batch item: [1, mel_bins, time]
            
            # Same for target
            if target_img.dim() == 2 and target_img.shape[0] == 1:
                time_frames = target_img.shape[1] // self.mel_bins
                target_img = target_img.reshape(1, self.mel_bins, time_frames)
                
            if target_img.dim() == 2:
                target_img = target_img.unsqueeze(0)
            elif target_img.dim() == 3 and target_img.shape[0] > 1:
                target_img = target_img[0].unsqueeze(0)
            
            # Normalize for better visualization
            pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-6)
            target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-6)
            
            # Log to TensorBoard
            if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image(
                    'pred_mel', pred_img, self.global_step, dataformats='CHW'
                )
                self.logger.experiment.add_image(
                    'target_mel', target_img, self.global_step, dataformats='CHW'
                )
        except Exception as e:
            # If logging fails, print info but don't crash training
            print(f"Warning: Failed to log spectrograms - {e}")
            if hasattr(pred_mel, 'shape'):
                print(f"Pred shape: {pred_mel.shape}")
            if hasattr(target_mel, 'shape'):
                print(f"Target shape: {target_mel.shape}")