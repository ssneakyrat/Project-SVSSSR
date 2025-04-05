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
    
    def compute_masked_loss(self, pred, target, mask=None):
        """
        Compute MSE loss with optional masking for variable-length data
        
        Args:
            pred: Predicted mel spectrogram [batch, channels, mel_bins, time]
            target: Target mel spectrogram [batch, channels, mel_bins, time]
            mask: Binary mask for valid time steps [batch, time]
            
        Returns:
            Loss value
        """
        if mask is None:
            return F.mse_loss(pred, target)
        
        # Expand mask to match spectrogram dimensions
        # [batch, time] -> [batch, 1, 1, time]
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Count valid time steps for averaging
        valid_steps = mask.sum()
        
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
            self._log_spectrograms(mel_pred[0], mel_ground_truth[0])
        
        return loss
    
    def _log_spectrograms(self, pred_mel, target_mel):
        """
        Log spectrograms to TensorBoard
        
        Args:
            pred_mel: Predicted mel spectrogram [channels, mel_bins, time]
            target_mel: Target mel spectrogram [channels, mel_bins, time]
        """
        # Convert to single-channel images
        pred_mel_img = pred_mel[0].unsqueeze(0)  # [1, mel_bins, time]
        target_mel_img = target_mel[0].unsqueeze(0)  # [1, mel_bins, time]
        
        # Log to TensorBoard
        if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_image(
                'pred_mel', pred_mel_img, self.global_step, dataformats='CHW'
            )
            self.logger.experiment.add_image(
                'target_mel', target_mel_img, self.global_step, dataformats='CHW'
            )