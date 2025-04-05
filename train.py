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
    """
    Load only the model weights from a checkpoint, ignoring optimizer and callback states.
    With improved error handling for architecture changes.
    """
    import torch
    import os
    
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
                
                # Print which modules are totally incompatible
                incompatible_modules = set()
                for param, _, _ in mismatch_params:
                    # Extract the module name (first part of parameter name)
                    module_name = param.split('.')[0]
                    incompatible_modules.add(module_name)
                
                print(f"Modules with incompatible parameters: {', '.join(incompatible_modules)}")
                print("Note: These modules will use random initialization")
            else:
                # No mismatches, load normally
                model.load_state_dict(state_dict)
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
    parser.add_argument('--load_weights', type=str, default=None, help='Path to checkpoint to load weights from (ignores training state)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume full training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length training mode')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs from config')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate from config')
    parser.add_argument('--disable_early_stopping', action='store_true', help='Disable early stopping callback')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (default depends on stage)')
    parser.add_argument('--freeze_earlier_stages', action='store_true', help='Freeze earlier stages when training in Stage 3')
    parser.add_argument('--accumulate_grad_batches', type=int, default=None, help='Accumulate gradients over N batches')
    parser.add_argument('--precision', type=str, default=None, choices=['32', '16-mixed'], help='Training precision')
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
        
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
        
    if args.freeze_earlier_stages:
        config['train']['freeze_earlier_stages'] = True
    
    current_stage = config['model']['current_stage']
    stage_epochs = config['model'].get('stage_epochs', [30, 30, 40])
    
    # Set appropriate defaults for Stage 3 if not specified
    if current_stage == 3 and not args.max_epochs and not args.learning_rate:
        # For Stage 3, longer training with lower learning rate by default
        if not args.learning_rate:
            original_lr = config['train']['learning_rate']
            config['train']['learning_rate'] = original_lr * 0.1
            print(f"Stage 3: Automatically reducing learning rate to {config['train']['learning_rate']} (1/10 of original)")
        
        if not args.batch_size:
            # Maybe reduce batch size for Stage 3 to fit in memory
            original_bs = config['train']['batch_size']
            if original_bs > 4:
                config['train']['batch_size'] = max(4, original_bs // 2)
                print(f"Stage 3: Automatically reducing batch size to {config['train']['batch_size']}")
    
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
    
    # Configure monitor interval based on stage
    monitor_interval = 1 if current_stage == 3 else 10
    
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
        if args.patience:
            patience = args.patience
        else:
            # Default stage-specific patience values that increase with stage complexity
            stage_patience_values = {
                1: 10,   # Stage 1: Basic patience 
                2: 20,   # Stage 2: More patience to handle transition
                3: 30,   # Stage 3: Even more patience for final refinement
            }
            patience = stage_patience_values.get(current_stage, 10)
        
        print(f"Using early stopping with patience of {patience} epochs")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)
    else:
        print("Early stopping has been disabled")
    
    try:
        # Initialize the model
        model = ProgressiveSVS(config)
        
        # If weights should be loaded from a previous stage checkpoint
        # but we want to start training fresh (with new optimizer and callbacks)
        if args.load_weights:
            print(f"Loading model weights from: {args.load_weights}")
            load_model_weights_only(model, args.load_weights)
            # Set resume to None to ensure we don't also try to resume training state
            args.resume = None
        
        data_module = SVSDataModule(config)
        
        trainer_kwargs = {
            'max_epochs': config['train']['num_epochs'],
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': monitor_interval,
            'accelerator': 'auto',
            'devices': 'auto',
        }
        
        # Set precision if specified
        if args.precision:
            trainer_kwargs['precision'] = args.precision
        
        # Set gradient accumulation if specified
        if args.accumulate_grad_batches:
            trainer_kwargs['accumulate_grad_batches'] = args.accumulate_grad_batches
            print(f"Accumulating gradients over {args.accumulate_grad_batches} batches")
        
        if 'gradient_clip_val' in config['train']:
            trainer_kwargs['gradient_clip_val'] = config['train']['gradient_clip_val']
        else:
            # Add gradient clipping by default for Stage 3
            if current_stage == 3:
                trainer_kwargs['gradient_clip_val'] = 1.0
                print("Stage 3: Enabling gradient clipping with value 1.0")
        
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, data_module, ckpt_path=args.resume)
        
        print(f"Training completed for stage {current_stage}.")
        
        if current_stage < 3:
            next_stage = current_stage + 1
            checkpoint_path = os.path.join(logger.log_dir, "checkpoints/last.ckpt")
            print(f"\nTo continue training with the next stage, run:")
            print(f"python train.py --stage {next_stage} --load_weights {checkpoint_path}")
        
        if current_stage == 3:
            print("\nStage 3 Training Recommendations:")
            print("1. If convergence is still poor, try:")
            print("   - Further reducing learning rate: --learning_rate 0.00005")
            print("   - Increasing gradient accumulation: --accumulate_grad_batches 4")
            print("   - Mixed precision training: --precision 16-mixed")
            print("   - Freezing earlier stages: --freeze_earlier_stages")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        H5FileManager.get_instance().close_all()
        print("H5 file resources released.")

if __name__ == "__main__":
    main()