import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from typing import Dict, Any, Optional

from models.svs_model import SVSModel
from models.base_model import BaseSVSModel
from data.dataset import SVSDataModule, H5FileManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SVS Model')
    
    # Basic configuration
    parser.add_argument('--config', type=str, default='config/svs_model.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # Training mode selection
    parser.add_argument('--mode', type=str, choices=['vae_only', 'diffusion_only', 'joint'], 
                        default=None, help='Training mode (overrides config)')
    
    # Data configuration
    parser.add_argument('--h5_path', type=str, default=None, 
                        help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None, 
                        help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size (overrides config)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None, 
                        help='Learning rate (overrides config)')
    parser.add_argument('--precision', type=str, choices=['32', '16-mixed'], default=None,
                        help='Precision for training (32 or 16-mixed)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Gradient clipping value')
    
    # Hardware configuration
    parser.add_argument('--gpus', type=int, default=None, 
                        help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=None, 
                        help='Number of dataloader workers')
    
    return parser.parse_args()


def apply_config_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Apply command line overrides to configuration."""
    # Basic configuration
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir
        
    # Training mode
    if args.mode:
        config['training']['mode'] = args.mode
        
    # Data configuration
    if args.h5_path:
        h5_dir = os.path.dirname(args.h5_path)
        h5_file = os.path.basename(args.h5_path)
        config['data']['dataset']['bin_dir'] = h5_dir
        config['data']['dataset']['bin_file'] = h5_file
    
    if args.data_key:
        config['data']['dataset']['data_key'] = args.data_key
    
    if args.max_samples:
        config['data']['dataset']['max_samples'] = args.max_samples
        
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        
    # Training configuration
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        
    if args.lr:
        config['training']['optimizer']['learning_rate'] = args.lr
        
    if args.precision:
        config['training']['precision'] = args.precision
        
    if args.clip_grad:
        config['training']['gradient_clip_val'] = args.clip_grad
        
    # Hardware configuration
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    
    return config


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the configuration."""
    logger.info("Configuration summary:")
    
    # Model configuration
    logger.info("Model configuration:")
    model_config = config["model"]
    logger.info(f"  - Base channels: {model_config['scale_factors']['base_channels']}")
    logger.info(f"  - Latent dimension: {model_config['scale_factors']['latent_dim']}")
    logger.info(f"  - Depth factor: {model_config['scale_factors']['depth_factor']}")
    logger.info(f"  - VAE channels: {model_config['vae']['encoder_channels']}")
    logger.info(f"  - Diffusion steps: {model_config['diffusion']['diffusion_steps']}")
    
    # Training configuration
    logger.info("Training configuration:")
    train_config = config["training"]
    logger.info(f"  - Mode: {train_config['mode']}")
    logger.info(f"  - Batch size: {train_config['batch_size']}")
    logger.info(f"  - Learning rate: {train_config['optimizer']['learning_rate']}")
    logger.info(f"  - Epochs: {train_config['num_epochs']}")
    logger.info(f"  - Precision: {train_config['precision']}")
    logger.info(f"  - Gradient clipping: {train_config['gradient_clip_val']}")
    
    # Data configuration
    logger.info("Data configuration:")
    data_config = config["data"]
    logger.info(f"  - H5 file: {os.path.join(data_config['dataset']['bin_dir'], data_config['dataset']['bin_file'])}")
    logger.info(f"  - Data key: {data_config['dataset']['data_key']}")
    logger.info(f"  - Max samples: {data_config['dataset']['max_samples']}")
    logger.info(f"  - Validation split: {data_config['dataset']['validation_split']}")


def create_callbacks(config: Dict[str, Any], save_dir: str) -> list:
    """Create training callbacks."""
    train_config = config["training"]
    
    callbacks = [
        # Save best model by validation loss
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True
        ),
        
        # Save best model specifically for the training mode
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            filename=f"best_{train_config['mode']}" + "-{epoch}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        ),
        
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
        
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=train_config.get("early_stop_patience", 20),
            mode="min",
            verbose=True
        )
    ]
    
    return callbacks


def create_trainer(config: Dict[str, Any], callbacks: list, save_dir: str) -> pl.Trainer:
    """Create PyTorch Lightning trainer."""
    train_config = config["training"]
    
    trainer_kwargs = {
        "max_epochs": train_config["num_epochs"],
        "callbacks": callbacks,
        "logger": TensorBoardLogger(save_dir=save_dir, name="logs"),
        "log_every_n_steps": train_config.get("log_interval", 10),
        "precision": train_config.get("precision", "16-mixed"),
        "gradient_clip_val": train_config.get("gradient_clip_val", 1.0),
        "deterministic": False,  # For better performance
        "accelerator": "auto",
        "devices": "auto",
        "accumulate_grad_batches": train_config.get("accumulate_grad_batches", 1)
    }
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer


def main():
    """Main training function."""
    try:
        # Enable tensor cores if available
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            torch.set_float32_matmul_precision('high')
            logger.info("Enabled high-precision tensor core operations")
        
        # Parse arguments
        args = parse_args()
        
        # Set random seed
        pl.seed_everything(args.seed)
        
        # Load configuration
        config = BaseSVSModel.load_config(args.config)
        
        # Apply command line overrides
        config = apply_config_overrides(config, args)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Set up directories
        save_dir = config["training"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        
        # Create model and data module
        model = SVSModel(config)
        data_module = SVSDataModule(config)
        
        # Create callbacks
        callbacks = create_callbacks(config, save_dir)
        
        # Create trainer
        trainer = create_trainer(config, callbacks, save_dir)
        
        # Train the model
        logger.info(f"Starting training in {config['training']['mode']} mode")
        trainer.fit(model, data_module, ckpt_path=args.resume)
        
        logger.info(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
    finally:
        # Clean up resources
        H5FileManager.get_instance().close_all()


if __name__ == "__main__":
    main()