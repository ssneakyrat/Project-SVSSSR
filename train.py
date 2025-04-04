import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import sys

sys.path.append('.')

from utils.utils import load_config
from models.progressive_svs import ProgressiveSVS
from data.dataset import SVSDataModule, check_dataset, fix_multiprocessing, H5FileManager

def main():
    parser = argparse.ArgumentParser(description='Train Progressive SVS Model')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--stage', type=int, default=None, help='Training stage (1, 2, or 3)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length training mode')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs from config')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    config = load_config(args.config)
    fix_multiprocessing()

    if args.stage:
        config['model']['current_stage'] = args.stage
    
    if args.variable_length:
        config['data']['variable_length'] = True
    
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    
    if args.max_epochs:
        config['train']['num_epochs'] = args.max_epochs
    
    current_stage = config['model']['current_stage']
    stage_epochs = config['model'].get('stage_epochs', [30, 30, 40])
    
    if not args.max_epochs and current_stage <= len(stage_epochs):
        config['train']['num_epochs'] = stage_epochs[current_stage-1]
    
    h5_path = os.path.join(config['data']['bin_dir'], config['data']['bin_file'])
    variable_length = config['data'].get('variable_length', True)
    
    if not check_dataset(h5_path, variable_length):
        print(f"Dataset check failed. Please ensure {h5_path} exists and has the required format.")
        print("You may need to run preprocess.py first.")
        return
    
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=f'stage{current_stage}',
        version=None
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename=f'svs-stage{current_stage}-' + '{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        ),
    ]
    
    try:
        model = ProgressiveSVS(config)
        data_module = SVSDataModule(config)
        
        trainer_kwargs = {
            'max_epochs': config['train']['num_epochs'],
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': 10,
            'accelerator': 'auto',
            'devices': 'auto',
        }
        
        if 'gradient_clip_val' in config['train']:
            trainer_kwargs['gradient_clip_val'] = config['train']['gradient_clip_val']
        
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, data_module, ckpt_path=args.resume)
        
        print(f"Training completed for stage {current_stage}.")
        
        if current_stage < 3:
            next_stage = current_stage + 1
            checkpoint_path = os.path.join(logger.log_dir, "checkpoints/last.ckpt")
            print(f"\nTo continue training with the next stage, run:")
            print(f"python train.py --stage {next_stage} --resume {checkpoint_path}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        H5FileManager.get_instance().close_all()
        print("H5 file resources released.")

if __name__ == "__main__":
    main()