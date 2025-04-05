import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.attention_free_svs import AttentionFreeSVS
from data.svs_dataset import SVSDataModule

def main():
    # Enable tensor cores on supported NVIDIA GPUs for better performance
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        # Setting to 'high' for best performance with minimal precision loss
        torch.set_float32_matmul_precision('high')
        print("Enabled high-precision tensor core operations for faster training")
    
    parser = argparse.ArgumentParser(description='Train Attention-Free SVS for mel spectrogram synthesis')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None, help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Add SVS-specific arguments
    parser.add_argument('--phone_key', type=str, default='phone_ids', help='Key for phone IDs in H5 file')
    parser.add_argument('--f0_key', type=str, default='f0', help='Key for F0 contours in H5 file')
    parser.add_argument('--duration_key', type=str, default='durations', help='Key for durations in H5 file')
    parser.add_argument('--midi_key', type=str, default='midi_ids', help='Key for MIDI IDs in H5 file')
    
    # Add arguments for variable length and audio length
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length processing')
    parser.add_argument('--max_audio_length', type=float, default=None, help='Maximum audio length in seconds')
    
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    # Apply command line overrides for data keys
    config['data']['phone_key'] = args.phone_key
    config['data']['f0_key'] = args.f0_key
    config['data']['duration_key'] = args.duration_key
    config['data']['midi_key'] = args.midi_key
    
    # Apply other command line overrides
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
    
    # Set variable length mode
    if args.variable_length:
        config['data']['variable_length'] = True
        config['model']['variable_length_mode'] = True
    
    # Set maximum audio length
    if args.max_audio_length:
        config['audio']['max_audio_length'] = args.max_audio_length
    elif 'max_audio_length' not in config['audio']:
        # Set default 10 second max if not specified
        config['audio']['max_audio_length'] = 10.0
    
    save_dir = args.save_dir or config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate and print the key model parameters
    phone_embed_dim = config['model']['phone_embed_dim']
    f0_embed_dim = config['model']['f0_embed_dim']
    duration_embed_dim = config['model']['duration_embed_dim']
    midi_embed_dim = config['model']['midi_embed_dim']
    
    backbone_channels = (
        config['model']['path1_channels'] + 
        config['model']['path2_channels'] + 
        config['model']['path3_channels']
    )
    
    total_params = (
        # Input encoders
        50 * phone_embed_dim +  # Phone embedding (assuming ~50 phones)
        (1 * f0_embed_dim + f0_embed_dim * f0_embed_dim) +  # F0 processing
        (1 * duration_embed_dim + duration_embed_dim * duration_embed_dim) +  # Duration processing
        128 * midi_embed_dim +  # MIDI embedding
        
        # Backbone (simplified estimate)
        (phone_embed_dim + f0_embed_dim + duration_embed_dim + midi_embed_dim) * backbone_channels +
        backbone_channels * backbone_channels * 3 +  # 3 paths
        
        # Decoder
        backbone_channels * backbone_channels * 2 +  # 2 upsampling layers
        backbone_channels * config['model']['mel_bins']  # Projection to mel
    )
    
    print(f"Configuration summary:")
    print(f"  - Input dimensions: {phone_embed_dim + f0_embed_dim + duration_embed_dim + midi_embed_dim}")
    print(f"  - Backbone channels: {backbone_channels}")
    print(f"  - Estimated parameters: ~{total_params/1000:.1f}K (~{total_params*4/1024/1024:.2f}MB)")
    print(f"  - Batch size: {config['train']['batch_size']}")
    print(f"  - Maximum audio length: {config['audio']['max_audio_length']} seconds")
    print(f"  - Variable length mode: {config['data'].get('variable_length', False)}")
    print(f"  - Data keys: {args.phone_key}, {args.f0_key}, {args.duration_key}, {args.midi_key}")
    
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='attention-free-svs-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
    ]
    
    model = AttentionFreeSVS(config)
    data_module = SVSDataModule(config)
    
    # Adjust trainer settings for GPU optimization
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['train'].get('log_interval', 10),
        deterministic=False,  # Set to False for better performance
        accelerator='auto',
        devices='auto',
        precision=config['train'].get('precision', '16-mixed'),
        accumulate_grad_batches=config['train'].get('accumulate_grad_batches', 1),
        gradient_clip_val=1.0  # Add gradient clipping for stability
    )
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume)
        print(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
    finally:
        from data.dataset import H5FileManager
        H5FileManager.get_instance().close_all()
    
if __name__ == "__main__":
    main()