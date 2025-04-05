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

def load_model_weights_only(model, checkpoint_path):
    """Load only the model weights from a checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            # Extract only the model weights
            state_dict = checkpoint['state_dict']
            
            # Check for any mismatching parameters
            model_state_dict = model.state_dict()
            mismatch_params = []
            
            for key in state_dict:
                if key in model_state_dict:
                    if state_dict[key].shape != model_state_dict[key].shape:
                        mismatch_params.append((key, state_dict[key].shape, model_state_dict[key].shape))
            
            if mismatch_params:
                print(f"Warning: Found {len(mismatch_params)} parameters with mismatched shapes")
                print("This indicates a model architecture change. Examples:")
                for i, (param, old_shape, new_shape) in enumerate(mismatch_params[:5]):
                    print(f"  {param}: checkpoint={old_shape}, current={new_shape}")
                
                if len(mismatch_params) > 5:
                    print(f"  ... and {len(mismatch_params) - 5} more mismatched parameters")
                
                print("\nAttempting to load compatible parameters only...")
                
                # Create a new state dict with only compatible parameters
                compatible_state_dict = {}
                for key, value in state_dict.items():
                    if key in model_state_dict and value.shape == model_state_dict[key].shape:
                        compatible_state_dict[key] = value
                
                # Load the compatible parameters
                model.load_state_dict(compatible_state_dict, strict=False)
                print(f"Successfully loaded {len(compatible_state_dict)}/{len(state_dict)} parameters")
                
            else:
                # No mismatches, load normally
                model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded all model weights from {checkpoint_path}")
            
            # Check the stage from the checkpoint
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                prev_config = checkpoint['hyper_parameters']['config']
                prev_stage = prev_config['model']['current_stage']
                print(f"Previous checkpoint was from stage {prev_stage}")
            
            return True
        else:
            print(f"Error: No state_dict found in checkpoint {checkpoint_path}")
            return False
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train Progressive SVS Model')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--stage', type=int, default=None, help='Training stage (1, 2, or 3)')
    parser.add_argument('--load_weights', type=str, default=None, help='Path to checkpoint to load weights from')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs from config')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--disable_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--freeze_earlier_stages', action='store_true', help='Freeze earlier stages when training in Stage 2 or 3')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    config = load_config(args.config)
    fix_multiprocessing()

    if args.stage:
        config['model']['current_stage'] = args.stage
    
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    
    if args.max_epochs:
        config['train']['num_epochs'] = args.max_epochs
        
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
        
    if args.freeze_earlier_stages:
        config['train']['freeze_earlier_stages'] = True
    
    # Get current stage and stage-specific settings
    current_stage = config['model']['current_stage']
    stage_epochs = config['model'].get('stage_epochs', [30, 30, 40])
    
    # Apply stage-specific configurations if not overridden
    if not args.max_epochs and current_stage <= len(stage_epochs):
        config['train']['num_epochs'] = stage_epochs[current_stage-1]
    
    # Stage-specific batch size and learning rate adjustments
    if current_stage == 2 and not args.batch_size:
        # Half the batch size for Stage 2
        original_bs = config['train']['batch_size']
        config['train']['batch_size'] = max(8, original_bs // 2)
    elif current_stage == 3 and not args.batch_size:
        # Quarter the batch size for Stage 3
        original_bs = config['train']['batch_size']
        config['train']['batch_size'] = max(4, original_bs // 4)
    
    if current_stage == 2 and not args.learning_rate:
        # Half the learning rate for Stage 2
        config['train']['learning_rate'] *= 0.5
    elif current_stage == 3 and not args.learning_rate:
        # Reduce learning rate by 10x for Stage 3
        config['train']['learning_rate'] *= 0.1
    
    # Verify dataset path
    h5_path = os.path.join(config['data']['bin_dir'], config['data']['bin_file'])
    variable_length = config['data'].get('variable_length', True)
    
    if not check_dataset(h5_path, variable_length):
        print(f"Dataset check failed. Please ensure {h5_path} exists and has the required format.")
        print("You may need to run preprocess.py first.")
        return
    
    # Setup logging and callbacks
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
            save_top_k=3 if current_stage == 3 else 1,  # Save top 3 models for Stage 3
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Setup early stopping with stage-appropriate patience
    if not args.disable_early_stopping:
        # Set different patience values based on stage
        patience = 10 * current_stage  # More patience for higher stages
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)
    
    try:
        # Initialize the model
        model = ProgressiveSVS(config)
        
        # Load weights from a previous stage if specified
        if args.load_weights:
            print(f"Loading model weights from: {args.load_weights}")
            load_model_weights_only(model, args.load_weights)
        
        data_module = SVSDataModule(config)
        
        # Configure training parameters
        trainer_kwargs = {
            'max_epochs': config['train']['num_epochs'],
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': 10,
            'accelerator': 'auto',
            'devices': 'auto',
        }
        
        # Use mixed precision if available to save memory
        if torch.cuda.is_available():
            trainer_kwargs['precision'] = '16-mixed'
        
        # Add gradient clipping for stability in Stage 3
        if current_stage == 3:
            trainer_kwargs['gradient_clip_val'] = 1.0
        
        # Accumulate gradients for larger effective batch size in Stage 3
        if current_stage == 3:
            trainer_kwargs['accumulate_grad_batches'] = 2
        
        trainer = pl.Trainer(**trainer_kwargs)
        
        # Start training
        trainer.fit(model, data_module, ckpt_path=args.resume)
        
        print(f"Training completed for stage {current_stage}.")
        
        if current_stage < 3:
            next_stage = current_stage + 1
            checkpoint_path = os.path.join(logger.log_dir, "checkpoints/last.ckpt")
            print(f"\nTo continue training with the next stage, run:")
            print(f"python train.py --stage {next_stage} --load_weights {checkpoint_path}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Clean up resources
        H5FileManager.get_instance().close_all()
        print("H5 file resources released.")

if __name__ == "__main__":
    main()