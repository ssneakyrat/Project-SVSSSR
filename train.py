import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.progressive_svs import ProgressiveSVS
from data.dataset import SVSDataModule, H5FileManager

def main():
    parser = argparse.ArgumentParser(description='Train Progressive SVS Model')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--stage', type=int, default=None, help='Training stage (1, 2, or 3)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    # Override current stage if specified
    if args.stage:
        config['model']['current_stage'] = args.stage
    
    # Get current stage
    current_stage = config['model']['current_stage']
    stage_epochs = config['model'].get('stage_epochs', [30, 30, 40])
    
    # Set epochs based on stage
    if current_stage <= len(stage_epochs):
        config['train']['num_epochs'] = stage_epochs[current_stage-1]
    
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=f'stage{current_stage}'
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
            mode='min'
        ),
    ]
    
    try:
        model = ProgressiveSVS(config)
        data_module = SVSDataModule(config)
        
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=10,
            accelerator='auto',
            devices='auto',
        )
        
        trainer.fit(model, data_module, ckpt_path=args.resume)
        
        print(f"Training completed for stage {current_stage}.")
        
        # Suggest moving to next stage if applicable
        if current_stage < 3:
            next_stage = current_stage + 1
            print(f"\nTo continue training with the next stage, run:")
            print(f"python train.py --stage {next_stage} --resume {save_dir}/stage{current_stage}/checkpoints/last.ckpt")
    finally:
        # Clean up H5 files
        H5FileManager.get_instance().close_all()

if __name__ == "__main__":
    main()