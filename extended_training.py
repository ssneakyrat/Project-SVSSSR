import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.attention_free_svs import AttentionFreeSVS
from data.svs_dataset import SVSDataModule
from model_visualizer import ModelVisualizer

class EnhancedAttentionFreeSVS(AttentionFreeSVS):
    """
    Extended version of AttentionFreeSVS with improved loss functions
    and visualization capabilities.
    """
    def __init__(self, config):
        super().__init__(config)
        self.visualizer = None
        self.use_spectral_loss = config.get('train', {}).get('use_spectral_loss', False)
        self.spectral_loss_weight = config.get('train', {}).get('spectral_loss_weight', 1.0)
        self.log_mel_outputs = config.get('train', {}).get('log_mel_outputs', True)
        
        # Remove tanh from decoder if specified
        if config.get('train', {}).get('remove_final_tanh', False):
            print("Removing tanh activation from decoder output")
            self.use_tanh = False
            # Replace final activation in decoder
            if hasattr(self.decoder, 'use_tanh'):
                self.decoder.use_tanh = False
    
    def initialize_visualizer(self, log_dir):
        """Initialize the visualizer for this model"""
        self.visualizer = ModelVisualizer(self, log_dir)
    
    def training_step(self, batch, batch_idx):
        """
        Execute a single training step with enhanced loss calculation
        """
        # Unpack the batch depending on whether using variable length
        if self.variable_length:
            phone_ids, f0, durations, midi_ids, mel_ground_truth, seq_mask, mel_mask = batch
        else:
            phone_ids, f0, durations, midi_ids, mel_ground_truth = batch
            seq_mask, mel_mask = None, None
        
        # Forward pass
        mel_pred = self(phone_ids, f0, durations, midi_ids)
        
        # Calculate primary MSE loss
        mse_loss = self.compute_masked_loss(mel_pred, mel_ground_truth, mel_mask)
        loss = mse_loss
        
        # Calculate spectral convergence loss if enabled
        if self.use_spectral_loss:
            spectral_loss = self.spectral_convergence_loss(mel_pred, mel_ground_truth, mel_mask)
            loss = mse_loss + self.spectral_loss_weight * spectral_loss
            
            # Log spectral loss
            self.log('train_spectral_loss', spectral_loss, prog_bar=True, sync_dist=True)
        
        # Log metrics
        self.log('train_mse_loss', mse_loss, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        # Visualize inputs/outputs occasionally
        if self.visualizer and batch_idx % 100 == 0:
            self.visualizer.log_inputs(phone_ids, f0, durations, midi_ids, self.global_step)
            self.visualizer.log_outputs(mel_pred, mel_ground_truth, self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Execute a single validation step with enhanced loss calculation
        """
        # Unpack the batch depending on whether using variable length
        if self.variable_length:
            phone_ids, f0, durations, midi_ids, mel_ground_truth, seq_mask, mel_mask = batch
        else:
            phone_ids, f0, durations, midi_ids, mel_ground_truth = batch
            seq_mask, mel_mask = None, None
        
        # Forward pass
        mel_pred = self(phone_ids, f0, durations, midi_ids)
        
        # Calculate primary MSE loss
        mse_loss = self.compute_masked_loss(mel_pred, mel_ground_truth, mel_mask)
        loss = mse_loss
        
        # Calculate spectral convergence loss if enabled
        if self.use_spectral_loss:
            spectral_loss = self.spectral_convergence_loss(mel_pred, mel_ground_truth, mel_mask)
            loss = mse_loss + self.spectral_loss_weight * spectral_loss
            
            # Log spectral loss
            self.log('val_spectral_loss', spectral_loss, prog_bar=True, sync_dist=True)
        
        # Log metrics
        self.log('val_mse_loss', mse_loss, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        # Log visualizations for the first batch
        if batch_idx == 0 and self.log_mel_outputs:
            # Adjust predicted mel to match target length for visualization
            adjusted_mel_pred = self.adjust_mel_length(mel_pred[0], mel_ground_truth[0].shape[-1])
            self._log_spectrograms(adjusted_mel_pred, mel_ground_truth[0])
            
            # Log with visualizer if available
            if self.visualizer:
                self.visualizer.log_inputs(phone_ids, f0, durations, midi_ids, self.global_step)
                self.visualizer.log_outputs(mel_pred, mel_ground_truth, self.global_step)
        
        return loss
    
    def spectral_convergence_loss(self, pred, target, mask=None):
        """
        Compute spectral convergence loss to better capture frequency content
        
        Args:
            pred: Predicted mel spectrogram [batch, mel_bins, time] or [batch, channels, mel_bins, time]
            target: Target mel spectrogram [batch, mel_bins, time] or [batch, channels, mel_bins, time]
            mask: Binary mask for valid time steps [batch, time]
            
        Returns:
            Spectral convergence loss
        """
        # Handle different tensor shapes
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [batch, mel_bins, time] -> [batch, 1, mel_bins, time]
        
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [batch, mel_bins, time] -> [batch, 1, mel_bins, time]
        
        # Adjust time dimension mismatch
        if pred.shape[-1] != target.shape[-1]:
            pred = self.adjust_mel_length(pred, target.shape[-1])
        
        # Compute Frobenius norm of the difference
        if mask is None:
            diff_norm = torch.norm(target - pred, p='fro', dim=(1, 2, 3))
            target_norm = torch.norm(target, p='fro', dim=(1, 2, 3))
            
            # Avoid division by zero
            eps = 1e-8
            loss = torch.mean(diff_norm / (target_norm + eps))
        else:
            # Expand mask to match spectrogram dimensions
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            
            # Compute masked norms
            masked_diff = (target - pred) * expanded_mask.float()
            masked_target = target * expanded_mask.float()
            
            diff_norm = torch.norm(masked_diff, p='fro', dim=(1, 2, 3))
            target_norm = torch.norm(masked_target, p='fro', dim=(1, 2, 3))
            
            # Avoid division by zero
            eps = 1e-8
            loss = torch.mean(diff_norm / (target_norm + eps))
        
        return loss


def main():
    parser = argparse.ArgumentParser(description='Extended training for Attention-Free SVS')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='extended_training', help='Directory to save models and logs')
    parser.add_argument('--use_spectral_loss', action='store_true', help='Enable spectral convergence loss')
    parser.add_argument('--spectral_loss_weight', type=float, default=0.1, help='Weight for spectral loss')
    parser.add_argument('--remove_tanh', action='store_true', help='Remove tanh activation from decoder')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    config['train']['num_epochs'] = args.epochs
    config['train']['batch_size'] = args.batch_size
    config['train']['learning_rate'] = args.learning_rate
    config['train']['save_dir'] = args.save_dir
    config['train']['use_spectral_loss'] = args.use_spectral_loss
    config['train']['spectral_loss_weight'] = args.spectral_loss_weight
    config['train']['remove_final_tanh'] = args.remove_tanh
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name='logs'
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, 'checkpoints'),
            filename='attention_free_svs-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Create model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = EnhancedAttentionFreeSVS.load_from_checkpoint(
            args.checkpoint,
            config=config
        )
    else:
        print("Creating new model")
        model = EnhancedAttentionFreeSVS(config)
    
    # Initialize visualizer
    model.initialize_visualizer(os.path.join(args.save_dir, 'visualizations'))
    
    # Create data module
    data_module = SVSDataModule(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        accelerator='auto',
        devices='auto',
        precision=config['train'].get('precision', '16-mixed'),
        accumulate_grad_batches=config['train'].get('accumulate_grad_batches', 1),
        gradient_clip_val=1.0
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    print(f"Training completed. Models saved to {args.save_dir}/checkpoints")
    print("Run TensorBoard to view results:")
    print(f"tensorboard --logdir={args.save_dir}/logs")


if __name__ == "__main__":
    main()