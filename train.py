# train.py
import os
import yaml
import torch
import argparse # Add argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Import custom modules
from data.dataset import DataModule # Assuming DataModule is in data/dataset.py
from models.progressive_svs import ProgressiveSVS # Import the new ProgressiveSVS model

# Update function signature to accept stage and ckpt_path
def train(stage, ckpt_path=None, config_path='config/model.yaml', freeze_weights=None): # Add freeze_weights parameter
    """
    Main training function.

    Args:
        stage (int): The current training stage (1, 2, or 3).
        ckpt_path (str, optional): Path to a checkpoint file for weight loading. Defaults to None.
        config_path (str): Path to the configuration YAML file.
    """
    # --- 1. Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return

    # --- Inject Stage into Config ---
    config['model']['current_stage'] = stage
    print(f"Set training stage to: {stage}")

    # --- Get Stage-Specific Epochs ---
    stage_key = f'stage{stage}'
    try:
        max_epochs_for_stage = config['train']['epochs_per_stage'][stage_key]
        print(f"Setting max_epochs for stage {stage} to: {max_epochs_for_stage}")
    except KeyError:
        print(f"Error: Epoch count for '{stage_key}' not found in config['train']['epochs_per_stage']. Check config/model.yaml.")
        return # Exit if the key is missing

    # --- Setup ---
    # Ensure log directories exist
    log_dir = config['train']['log_dir']
    tb_log_dir = os.path.join(log_dir, config['train']['tensorboard_log_dir'])
    ckpt_dir = os.path.join(log_dir, config['train']['checkpoint_dir'])
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    print(f"Model checkpoints will be saved to: {ckpt_dir}")

    # --- 2. Initialize DataModule ---
    print("Initializing DataModule...")
    try:
        data_module = DataModule(config)
        # Run setup to prepare datasets (train/val split) AND read vocab_size
        data_module.setup()
        print("DataModule setup completed.")

        # Check if vocab_size was read successfully
        if data_module.vocab_size is None:
             print("Error: vocab_size could not be determined from the dataset.")
             return # Stop execution if vocab_size is missing

        # Update config with the dynamically read vocab_size for the model
        config['model']['vocab_size'] = data_module.vocab_size
        print(f"Using dynamically determined vocab_size: {config['model']['vocab_size']}")

    except FileNotFoundError as e:
         # More specific error message for missing HDF5 file
         h5_path_expected = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
         print(f"Error: HDF5 dataset file not found at '{h5_path_expected}'. Did you run preprocessing?")
         print(f"Original error: {e}")
         return
    except ValueError as e: # Catch the ValueError raised if vocab_size attr is missing
        print(f"Error during DataModule setup: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during DataModule initialization/setup: {e}")
        # Optionally re-raise for more detailed traceback: raise e
        return


    # --- 3. Initialize Model ---
    print(f"Initializing {config['model']['name']} model...") # Use name from config
    model = ProgressiveSVS(config)
    print("Model initialized successfully.")

    # --- Load Weights if Checkpoint Provided ---
    if ckpt_path:
        # If ckpt_path is relative, make it absolute based on current dir for loading
        ckpt_path_abs = os.path.abspath(ckpt_path)
        if os.path.exists(ckpt_path_abs):
            print(f"Loading weights from checkpoint: {ckpt_path_abs}") # Log absolute path
            try:
                checkpoint = torch.load(ckpt_path_abs, map_location='cpu')
                # Load weights, allowing for missing/extra keys (e.g., loading stage 1 into stage 2)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("Weights loaded successfully.")

                # --- Freeze Weights if Requested ---
                if freeze_weights:
                    print("Freezing weights for LowResModel and MidResUpsampler...")
                    frozen_count = 0
                    for name, param in model.named_parameters():
                        if name.startswith('low_res_model.') or name.startswith('mid_res_upsampler.'):
                            param.requires_grad = False
                            frozen_count += 1
                    print(f"Froze {frozen_count} parameters in LowResModel and MidResUpsampler.")
                else:
                    print("Loaded weights remain trainable.")

                print("Weights loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint weights: {e}")
                print("Proceeding without loaded weights.")
        else:
            print(f"Warning: Checkpoint path specified ('{ckpt_path}', resolved to '{ckpt_path_abs}') but not found. Starting from scratch.")

    # --- 4. Configure Callbacks and Logger ---
    print("Configuring logger and callbacks...")
    # TensorBoard Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config['train']['tensorboard_log_dir'],
        version=None # Auto-increment version
    )

    # Model Checkpointing
    # Saves the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, # This should be relative or absolute as defined in config
        filename=f'progressive-svs-stage{stage}-{{epoch:02d}}-{{val_loss:.2f}}', # Include stage in filename
        save_top_k=1,          # Save only the best model
        monitor='val_loss',    # Monitor validation loss
        mode='min',            # Mode should be 'min' for loss
        save_last=True         # Optionally save the last checkpoint as well
    )
    print("Logger and callbacks configured.")

    # --- 5. Initialize Trainer ---
    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs_for_stage,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None, # Specify number of devices
        log_every_n_steps=config['train'].get('log_interval', 50), # How often to log within an epoch
        check_val_every_n_epoch=config['train']['val_interval'],
        # Add other trainer flags as needed, e.g.:
        # deterministic=True,
        gradient_clip_val=config['train'].get('gradient_clip_val'), # Use value from config
    )
    print(f"Trainer initialized. Using accelerator: {trainer.accelerator}")

    # --- 6. Start Training ---
    print("Starting training...")
    try:
        # Do NOT pass ckpt_path here, as we handle weight loading manually above
        trainer.fit(model, datamodule=data_module)
        print("Training finished.")

        # --- Print Next Step / Completion ---
        best_ckpt_path_abs = checkpoint_callback.best_model_path
        if best_ckpt_path_abs and os.path.exists(best_ckpt_path_abs): # Check if path exists
            # Convert absolute path to relative path from current directory
            best_ckpt_path_rel = os.path.relpath(best_ckpt_path_abs, start=os.curdir)
            if stage < 3: # Assuming 3 stages total
                print(f"\nStage {stage} complete. To run stage {stage + 1}, use:")
                # Use the relative path in the command
                print(f"python train.py --stage {stage + 1} --ckpt {best_ckpt_path_rel}")
            else:
                print(f"\nStage {stage} (Final) complete.")
                # Print the relative path
                print(f"Best checkpoint saved at: {best_ckpt_path_rel}")
        else:
            print("\nTraining finished, but no best checkpoint path found or saved (check ModelCheckpoint config and disk space).")
            # Optionally print the last checkpoint path if it exists (also make relative)
            last_ckpt_path_abs = checkpoint_callback.last_model_path
            if last_ckpt_path_abs and os.path.exists(last_ckpt_path_abs):
                 last_ckpt_path_rel = os.path.relpath(last_ckpt_path_abs, start=os.curdir)
                 print(f"Last checkpoint saved at: {last_ckpt_path_rel}")


    except Exception as e:
        import traceback
        print(f"An error occurred during training: {e}")
        print("--- Full Traceback ---")
        traceback.print_exc() # Print the full traceback
        print("----------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ProgressiveSVS model.")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Training stage (1, 2, or 3).')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint file to load weights from (optional). Can be relative or absolute.')
    parser.add_argument('--config', type=str, default='config/model.yaml',
                        help='Path to the configuration YAML file.')
    parser.add_argument('--freeze_weights', type=str, default=None, choices=['true', 'false'],
                        help='Override config setting for freezing loaded weights (true/false).')
    args = parser.parse_args()

    # Set random seed for reproducibility (optional but recommended)
    pl.seed_everything(42, workers=True)

    # Determine final freeze setting (CLI overrides config)
    freeze_setting = None
    if args.freeze_weights is not None:
        freeze_setting = args.freeze_weights.lower() == 'true'
        print(f"Freeze weights override from CLI: {freeze_setting}")
    # Note: The config value will be read inside the train function if freeze_setting is None here

    # Pass potentially relative ckpt path; it will be made absolute inside train() if needed for loading
    train(stage=args.stage, ckpt_path=args.ckpt, config_path=args.config, freeze_weights=freeze_setting)