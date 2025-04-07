# train_vocoder.py

import os
import argparse
import torch
import yaml
import logging
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import DataModule
from models.vocoder import VocoderModel
from utils.utils import setup_logging

logger = logging.getLogger(__name__)

def main(args):
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command-line arguments if provided
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.epochs:
        config['train']['vocoder_max_epochs'] = args.epochs
    if args.learning_rate:
        config['train']['vocoder_learning_rate'] = args.learning_rate
    
    # Setup checkpoint directory and logging
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"vocoder_{args.name}_{current_time}" if args.name else f"vocoder_{current_time}"
    
    # Logging directories
    log_dir = os.path.join(config['train']['log_dir'], 'vocoder', run_name)
    tensorboard_dir = os.path.join(config['train']['tensorboard_log_dir'], 'vocoder', run_name)
    checkpoint_dir = os.path.join(config['train']['checkpoint_dir'], 'vocoder', run_name)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(log_dir, 'training.log')
    setup_logging(level=logging.INFO, log_file=log_file)
    
    # Log configuration for reference
    logger.info(f"Run name: {run_name}")
    logger.info(f"Tensorboard dir: {tensorboard_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    
    # Save configuration to the run directory
    config_save_path = os.path.join(log_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_save_path}")
    
    # Initialize data module
    logger.info("Initializing DataModule")
    data_module = DataModule(config)
    
    # Initialize model
    logger.info("Initializing VocoderModel")
    model = VocoderModel(config)
    
    # Load from checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = VocoderModel.load_from_checkpoint(args.checkpoint, config=config)
    
    # Initialize callbacks
    callbacks = [
        # Checkpoint callback
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
            save_last=True
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Add early stopping if enabled
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                patience=args.patience,
                mode='min',
                verbose=True
            )
        )
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.dirname(tensorboard_dir),
        name=os.path.basename(tensorboard_dir),
        default_hp_metric=False
    )
    
    # Training precision
    precision = '16-mixed' if args.fp16 else '32-true'
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['vocoder_max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        logger=tb_logger,
        callbacks=callbacks,
        #log_every_n_steps=config['train']['log_interval'],
        #val_check_interval=config['train']['log_vocoder_audio_epoch_interval'], #1.0,  # Validate once per epoch
        precision=precision,
        gradient_clip_val=config['train'].get('gradient_clip_val', 0.5),
        accumulate_grad_batches=args.gradient_accumulation
    )
    
    # Start training
    logger.info("Starting training")
    trainer.fit(model, data_module)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_checkpoint_path)
    logger.info(f"Final model saved to {final_checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny WaveRNN Vocoder")
    
    # Configuration and model options
    parser.add_argument('--config', type=str, default='config/model.yaml',
                        help='Path to configuration file')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for the training run')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--gradient_accumulation', type=int, default=2,
                        help='Gradient accumulation steps')
    
    # Hardware/performance options
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    
    # Early stopping options
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Setup basic logging for command-line output
    setup_logging(level=logging.INFO)
    
    # Call main function
    main(args)