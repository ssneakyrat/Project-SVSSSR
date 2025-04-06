# train.py
import os
import yaml
import torch
import argparse # Add argparse
import sys      # Add sys for checking command-line arguments
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
    # --- Determine Effective Freeze Setting ---
    # This needs to be determined BEFORE model initialization
    effective_freeze = freeze_weights # Start with the argument value (could be True, False, or None)
    if effective_freeze is None:
        # Argument not provided via CLI, check config default
        config_default_freeze = config.get('train', {}).get('freeze_loaded_weights', False) # Default to False if keys missing
        effective_freeze = config_default_freeze
        print(f"Freeze weights setting derived: {effective_freeze} (from CLI arg or config default)")
    else:
        print(f"Freeze weights setting provided via CLI: {effective_freeze}")

    # --- Initialize Model ---
    print(f"Initializing {config['model']['name']} model...") # Use name from config
    model = ProgressiveSVS(config) # Initialize normally, hooks removed from model
    print("Model initialized successfully.")

    # --- Manual Checkpoint Loading & State Setup ---
    optimizer = None
    scheduler_config = None # Will hold the dict like {'scheduler': ..., 'monitor': ...}
    loaded_checkpoint_data = None

    if ckpt_path:
        ckpt_path_abs = os.path.abspath(ckpt_path)
        if os.path.exists(ckpt_path_abs):
            print(f"--- Attempting to load checkpoint: {ckpt_path_abs} ---")
            try:
                # Load Full Checkpoint Dictionary (CPU first)
                loaded_checkpoint_data = torch.load(ckpt_path_abs, map_location='cpu')
                print("Checkpoint dictionary loaded.")

                # Load Model Weights
                print("Loading model state_dict...")
                model.load_state_dict(loaded_checkpoint_data['state_dict'], strict=False)
                print("Model state_dict loaded.")

                # Apply Manual Freezing (AFTER loading weights)
                if effective_freeze and stage > 1:
                    print(f"Applying manual freezing for stage {stage}...")
                    frozen_count = 0
                    # Freeze LowResModel
                    if hasattr(model, 'low_res_model'):
                        for name, param in model.low_res_model.named_parameters():
                            if param.requires_grad:
                                param.requires_grad = False
                                frozen_count += 1
                    # Freeze MidResUpsampler for Stage 3
                    if stage == 3 and hasattr(model, 'mid_res_upsampler'):
                        for name, param in model.mid_res_upsampler.named_parameters():
                            if param.requires_grad:
                                param.requires_grad = False
                                frozen_count += 1
                    print(f"Manually froze {frozen_count} parameters.")
                else:
                     print("Freezing not applied (Stage 1 or effective_freeze=False).")

                # Configure Optimizer/Scheduler (AFTER freezing)
                print("Configuring optimizer and scheduler based on current model state...")
                optimizer_conf_output = model.configure_optimizers()
                if isinstance(optimizer_conf_output, dict):
                    optimizer = optimizer_conf_output['optimizer']
                    scheduler_config = optimizer_conf_output.get('lr_scheduler')
                elif isinstance(optimizer_conf_output, torch.optim.Optimizer):
                    optimizer = optimizer_conf_output
                    scheduler_config = None
                # TODO: Add handling for tuple return type if necessary
                else:
                    raise TypeError(f"Unsupported return type from configure_optimizers: {type(optimizer_conf_output)}")
                print("Optimizer and scheduler configured.")

                # Load Optimizer State
                if 'optimizer_states' in loaded_checkpoint_data and optimizer:
                    print("Loading optimizer state...")
                    try:
                        optimizer.load_state_dict(loaded_checkpoint_data['optimizer_states'][0])
                        print("Optimizer state loaded.")
                    except ValueError as e:
                        print(f"Warning: Optimizer state mismatch. Could not load state: {e}. Optimizer starts fresh.")
                    except Exception as e:
                        print(f"Error loading optimizer state: {e}. Optimizer starts fresh.")
                elif optimizer:
                    print("Optimizer state not found in checkpoint. Optimizer starts fresh.")

                # Load Scheduler State
                if 'lr_schedulers' in loaded_checkpoint_data and scheduler_config:
                    print("Loading LR scheduler state...")
                    try:
                        scheduler_instance = scheduler_config['scheduler']
                        scheduler_instance.load_state_dict(loaded_checkpoint_data['lr_schedulers'][0])
                        print("LR scheduler state loaded.")
                    except Exception as e:
                        print(f"Error loading LR scheduler state: {e}. Scheduler starts fresh.")
                elif scheduler_config:
                    print("LR scheduler state not found in checkpoint. Scheduler starts fresh.")

            except Exception as e:
                print(f"ERROR during checkpoint loading/processing: {e}")
                print("Proceeding without loaded state (starting from scratch).")
                loaded_checkpoint_data = None # Ensure flag is reset
                optimizer = None
                scheduler_config = None
        else:
            print(f"Warning: Checkpoint path specified ('{ckpt_path}') but not found. Starting from scratch.")
            ckpt_path = None # Set to None if path doesn't exist

    # Configure Fresh Optimizer/Scheduler if no checkpoint loaded successfully
    if not loaded_checkpoint_data:
        print("Configuring fresh optimizer and scheduler...")
        try:
            optimizer_conf_output = model.configure_optimizers()
            if isinstance(optimizer_conf_output, dict):
                optimizer = optimizer_conf_output['optimizer']
                scheduler_config = optimizer_conf_output.get('lr_scheduler')
            elif isinstance(optimizer_conf_output, torch.optim.Optimizer):
                optimizer = optimizer_conf_output
                scheduler_config = None
            # TODO: Add handling for tuple return type if necessary
            else:
                raise TypeError(f"Unsupported return type from configure_optimizers: {type(optimizer_conf_output)}")
            print("Fresh optimizer and scheduler configured.")
        except Exception as e:
            print(f"Error configuring fresh optimizer/scheduler: {e}. Cannot proceed.")
            return None # Critical error


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
        # --- Manually Assign Optimizer/Scheduler to Trainer ---
        if optimizer:
            trainer.optimizers = [optimizer]
            print("Optimizer manually assigned to Trainer.")
        else:
            print("ERROR: Optimizer was not configured. Cannot train.")
            return None

        if scheduler_config:
            # Ensure it's a list containing the scheduler config dictionary
            trainer.lr_schedulers = [scheduler_config]
            print("LR Scheduler manually assigned to Trainer.")
        else:
            trainer.lr_schedulers = [] # Ensure it's an empty list if no scheduler
            print("No LR Scheduler to assign to Trainer.")

        # --- Start Training ---
        # Do NOT pass ckpt_path here, loading is handled manually above
        trainer.fit(model, datamodule=data_module)
        print("Training finished.")

        # --- Determine and Return Checkpoint Path ---
        best_ckpt_path_rel = None
        last_ckpt_path_rel = None

        # Check for best checkpoint
        best_ckpt_path_abs = checkpoint_callback.best_model_path
        if best_ckpt_path_abs and os.path.exists(best_ckpt_path_abs):
            best_ckpt_path_rel = os.path.relpath(best_ckpt_path_abs, start=os.curdir)
            print(f"\nStage {stage} finished. Best checkpoint saved at: {best_ckpt_path_rel}")
            # Print next step info only if running manually (not sequential)
            # We'll handle this logic in the main block now
            # if stage < 3:
            #     print(f"To run stage {stage + 1}, use: python train.py --stage {stage + 1} --ckpt {best_ckpt_path_rel}")

        # Check for last checkpoint if best wasn't found or saved
        last_ckpt_path_abs = checkpoint_callback.last_model_path
        if last_ckpt_path_abs and os.path.exists(last_ckpt_path_abs):
            last_ckpt_path_rel = os.path.relpath(last_ckpt_path_abs, start=os.curdir)
            if not best_ckpt_path_rel: # Only print if best wasn't found
                 print(f"\nStage {stage} finished. Last checkpoint saved at: {last_ckpt_path_rel}")

        # Return best if available, otherwise last, otherwise None
        if best_ckpt_path_rel:
            return best_ckpt_path_rel
        elif last_ckpt_path_rel:
             print("Warning: Best checkpoint not found or saved, using last checkpoint for next stage.")
             return last_ckpt_path_rel
        else:
            print("\nTraining finished, but no usable checkpoint (best or last) found.")
            return None

    except Exception as e:
        import traceback
        print(f"An error occurred during training stage {stage}: {e}")
        print("--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
        return None # Return None if training failed
if __name__ == "__main__":
    # Set random seed early for reproducibility
    pl.seed_everything(42, workers=True)

    # Default config path
    default_config_path = 'config/model.yaml'

    # Check if arguments were provided
    if len(sys.argv) > 1:
        # --- Argument-based Training (Existing Behavior) ---
        print("Arguments detected, running single stage training.")
        parser = argparse.ArgumentParser(description="Train the ProgressiveSVS model for a specific stage.")
        parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                            help='Training stage (1, 2, or 3).')
        parser.add_argument('--ckpt', type=str, default=None,
                            help='Path to checkpoint file to load weights from (optional). Can be relative or absolute.')
        parser.add_argument('--config', type=str, default=default_config_path,
                            help=f'Path to the configuration YAML file (default: {default_config_path}).')
        parser.add_argument('--freeze_weights', type=str, default=None, choices=['true', 'false'],
                            help='Override config setting for freezing loaded weights (true/false).')
        args = parser.parse_args()

        # Determine final freeze setting (CLI overrides config)
        freeze_setting = None
        if args.freeze_weights is not None:
            freeze_setting = args.freeze_weights.lower() == 'true'
            print(f"Freeze weights override from CLI: {freeze_setting}")

        # Run the single specified stage
        # The return value (checkpoint path) is not needed here as the function prints info
        train(stage=args.stage, ckpt_path=args.ckpt, config_path=args.config, freeze_weights=freeze_setting)
        print("\nSingle stage training finished.")

    else:
        # --- Sequential Training (New Behavior) ---
        print("No arguments provided. Starting sequential training (Stages 1-3)...")
        current_ckpt_path = None
        freeze_next_stage = False # Stage 1 does not freeze weights
        config_path = default_config_path # Use default config

        for stage in range(1, 4): # Loop through stages 1, 2, 3
            print(f"\n--- Starting Sequential Training: Stage {stage}/3 ---")
            print(f"Using config: {config_path}")
            if current_ckpt_path:
                print(f"Loading checkpoint: {current_ckpt_path}")
            print(f"Freeze loaded weights: {freeze_next_stage}")

            # Call the train function for the current stage
            next_ckpt_path = train(
                stage=stage,
                ckpt_path=current_ckpt_path,
                config_path=config_path,
                freeze_weights=freeze_next_stage
            )

            # Check if the stage completed successfully and produced a checkpoint
            if next_ckpt_path is None:
                print(f"\n--- ERROR: Stage {stage} failed or did not produce a checkpoint. Stopping sequential training. ---")
                break # Exit the loop if a stage fails

            # Prepare for the next stage
            current_ckpt_path = next_ckpt_path
            freeze_next_stage = True # Freeze weights for stages 2 and 3

            # Check if this was the last stage
            if stage == 3:
                print("\n--- Sequential Training (Stages 1-3) Completed Successfully! ---")
                print(f"Final checkpoint saved at: {current_ckpt_path}")

        # This else block executes if the loop completes without break
        # else:
        #     print("\n--- Sequential Training (Stages 1-3) Completed Successfully! ---")
        #     # The final checkpoint path is already printed in the last iteration