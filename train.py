import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.svs_model import SVSModel
from data.dataset import SVSDataModule, H5FileManager


def main():
    # Enable tensor cores if available for faster training
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        torch.set_float32_matmul_precision('high')
        print("Enabled high-precision tensor core operations for faster training")
    
    parser = argparse.ArgumentParser(description='Train Latent Diffusion SVS Model')
    parser.add_argument('--config', type=str, default='config/diffusion_model.yaml', help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None, help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Add tensor core precision argument
    parser.add_argument('--precision', type=str, choices=['32', '16-mixed'], default='16-mixed',
                        help='Precision for training (32 or 16-mixed)')
    
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    
    if args.h5_path:
        h5_dir = os.path.dirname(args.h5_path)
        h5_file = os.path.basename(args.h5_path)
        config['data']['bin_dir'] = h5_dir
        config['data']['bin_file'] = h5_file
    
    if args.data_key:
        config['data']['data_key'] = args.data_key
    
    if args.max_samples:
        config['data']['max_samples'] = args.max_samples
    
    if args.precision:
        config['train']['precision'] = args.precision
    
    save_dir = args.save_dir or config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Print configuration summary
    print(f"Configuration summary:")
    print(f"  - Maximum audio length: {config['audio']['max_audio_length']} seconds")
    print(f"  - Batch size: {config['train']['batch_size']}")
    print(f"  - Diffusion steps: {config['model']['diffusion']['diffusion_steps']}")
    print(f"  - Precision: {config['train']['precision']}")
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='svs-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min'
        ),
    ]
    
    # Create model and data module
    model = SVSModel(config)
    data_module = SVSDataModule(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['train'].get('log_interval', 10),
        deterministic=False,  # Set to False for better performance
        accelerator='auto',
        devices='auto',
        precision=config['train'].get('precision', '16-mixed'),
        gradient_clip_val=1.0,  # Add gradient clipping for stability
        accumulate_grad_batches=config['train'].get('accumulate_grad_batches', 1)
    )
    
    try:
        # Train model
        trainer.fit(model, data_module, ckpt_path=args.resume)
        print(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
    finally:
        # Close H5 files
        H5FileManager.get_instance().close_all()
    
if __name__ == "__main__":
    main()