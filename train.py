# train.py
import os
import yaml
import torch
import argparse # Add argparse
import sys      # Add sys for checking command-line arguments
import logging  # Import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Import custom modules
from data.dataset import DataModule # Assuming DataModule is in data/dataset.py
from models.progressive_svs import ProgressiveSVS # Import the new ProgressiveSVS model
from utils.utils import setup_logging # Import the logging setup function

logger = logging.getLogger(__name__) # Get logger instance

# Update function signature to accept stage and ckpt_path
def train(stage, ckpt_path=None, config_path='config/model.yaml'): # Removed freeze_weights parameter
    """
    Main training function.

    Args:
        stage (int): The current training stage (1, 2, or 3).
        ckpt_path (str, optional): Path to a checkpoint file for weight loading. Defaults to None.
        config_path (str): Path to the configuration YAML file.
    """
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return

    # --- Inject Stage into Config ---
    config['model']['current_stage'] = stage
    logger.info(f"Set training stage to: {stage}")

    # --- Get Stage-Specific Epochs ---
    stage_key = f'stage{stage}'
    try:
        max_epochs_for_stage = config['train']['epochs_per_stage'][stage_key]
        logger.info(f"Setting max_epochs for stage {stage} to: {max_epochs_for_stage}")
    except KeyError:
        logger.error(f"Epoch count for '{stage_key}' not found in config['train']['epochs_per_stage']. Check config/model.yaml.")
        return # Exit if the key is missing

    # --- Setup ---
    # Ensure log directories exist
    log_dir = config['train']['log_dir']
    tb_log_dir = os.path.join(log_dir, config['train']['tensorboard_log_dir'])
    ckpt_dir = os.path.join(log_dir, config['train']['checkpoint_dir'])
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logger.info(f"Model checkpoints will be saved to: {ckpt_dir}")

    # --- 2. Initialize DataModule ---
    logger.info("Initializing DataModule...")
    try:
        data_module = DataModule(config)
        # Run setup to prepare datasets (train/val split) AND read vocab_size
        data_module.setup()
        logger.info("DataModule setup completed.")

        # Check if vocab_size was read successfully
        if data_module.vocab_size is None:
             logger.error("vocab_size could not be determined from the dataset.")
             return # Stop execution if vocab_size is missing

        # Update config with the dynamically read vocab_size for the model
        config['model']['vocab_size'] = data_module.vocab_size
        logger.info(f"Using dynamically determined vocab_size: {config['model']['vocab_size']}")

    except FileNotFoundError as e:
         # More specific error message for missing HDF5 file
         h5_path_expected = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
         logger.error(f"HDF5 dataset file not found at '{h5_path_expected}'. Did you run preprocessing?")
         logger.error(f"Original error: {e}")
         return
    except ValueError as e: # Catch the ValueError raised if vocab_size attr is missing
        logger.error(f"Error during DataModule setup: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during DataModule initialization/setup: {e}", exc_info=True)
        # Optionally re-raise for more detailed traceback: raise e
        return


    # --- 3. Initialize Model ---
    # Freezing logic will be handled within the model based on the stage and checkpoint loading
    # --- Initialize Model ---
    logger.info(f"Initializing {config['model']['name']} model...") # Use name from config
    model = ProgressiveSVS(config) # Initialize normally, hooks removed from model
    logger.info("Model initialized successfully.")

    # Model initialization is done. Checkpoint loading and state restoration
    # will be handled by PyTorch Lightning via trainer.fit(ckpt_path=...)

    # --- 4. Configure Callbacks and Logger ---
    logger.info("Configuring logger and callbacks...")
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
    logger.info("Logger and callbacks configured.")

    # --- 5. Initialize Trainer ---
    logger.info("Initializing PyTorch Lightning Trainer...")
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
    logger.info(f"Trainer initialized. Using accelerator: {trainer.accelerator}")

    # --- 6. Start Training ---
    logger.info("Starting training...")
    try:
        # Optimizer and scheduler configuration/assignment is handled internally by Lightning
        # when resuming from a checkpoint or starting fresh.
        # --- Start Training ---
        # Pass ckpt_path to trainer.fit to let Lightning handle resuming
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        logger.info("Training finished.")

        # --- Determine and Return Checkpoint Path ---
        best_ckpt_path_rel = None
        last_ckpt_path_rel = None

        # Check for best checkpoint
        best_ckpt_path_abs = checkpoint_callback.best_model_path
        if best_ckpt_path_abs and os.path.exists(best_ckpt_path_abs):
            best_ckpt_path_rel = os.path.relpath(best_ckpt_path_abs, start=os.curdir)
            logger.info(f"Stage {stage} finished. Best checkpoint saved at: {best_ckpt_path_rel}")
            # Print next step info only if running manually (not sequential)
            # We'll handle this logic in the main block now
            # if stage < 3:
            #     print(f"To run stage {stage + 1}, use: python train.py --stage {stage + 1} --ckpt {best_ckpt_path_rel}")

        # Check for last checkpoint if best wasn't found or saved
        last_ckpt_path_abs = checkpoint_callback.last_model_path
        if last_ckpt_path_abs and os.path.exists(last_ckpt_path_abs):
            last_ckpt_path_rel = os.path.relpath(last_ckpt_path_abs, start=os.curdir)
            if not best_ckpt_path_rel: # Only print if best wasn't found
                 logger.info(f"Stage {stage} finished. Last checkpoint saved at: {last_ckpt_path_rel}")

        # Return best if available, otherwise last, otherwise None
        if best_ckpt_path_rel:
            return best_ckpt_path_rel
        elif last_ckpt_path_rel:
             logger.warning("Best checkpoint not found or saved, using last checkpoint for next stage.")
             return last_ckpt_path_rel
        else:
            logger.warning("Training finished, but no usable checkpoint (best or last) found.")
            return None

    except Exception as e:
        import traceback
        # Use logger.exception to include traceback automatically
        logger.exception(f"An error occurred during training stage {stage}: {e}")
        return None # Return None if training failed
if __name__ == "__main__":
    # --- Setup Logging ---
    # TODO: Consider making log level and file configurable via args or config file
    setup_logging(level=logging.INFO) # Setup logging before anything else
    logger.info("Starting training script...")

    # Set random seed early for reproducibility
    pl.seed_everything(42, workers=True)
    logger.info("Set random seed to 42.")

    # Default config path
    default_config_path = 'config/model.yaml'

    # Check if arguments were provided
    if len(sys.argv) > 1:
        # --- Argument-based Training (Existing Behavior) ---
        logger.info("Arguments detected, running single stage training.")
        parser = argparse.ArgumentParser(description="Train the ProgressiveSVS model for a specific stage.")
        parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                            help='Training stage (1, 2, or 3).')
        parser.add_argument('--ckpt', type=str, default=None,
                            help='Path to checkpoint file to load weights from (optional). Can be relative or absolute.')
        parser.add_argument('--config', type=str, default=default_config_path,
                            help=f'Path to the configuration YAML file (default: {default_config_path}).')
                            # help='Override config setting for freezing loaded weights (true/false).') # Removed freeze_weights arg
        args = parser.parse_args()

        # Removed freeze_setting logic

        # Run the single specified stage
        # The return value (checkpoint path) is not needed here as the function prints info
        train(stage=args.stage, ckpt_path=args.ckpt, config_path=args.config) # Removed freeze_weights
        logger.info("Single stage training finished.")

    else:
        # --- Sequential Training (New Behavior) ---
        logger.info("No arguments provided. Starting sequential training (Stages 1-3)...")
        current_ckpt_path = None
        # freeze_next_stage = False # Removed freeze logic
        config_path = default_config_path # Use default config

        for stage in range(1, 4): # Loop through stages 1, 2, 3
            logger.info(f"--- Starting Sequential Training: Stage {stage}/3 ---")
            logger.info(f"Using config: {config_path}")
            if current_ckpt_path:
                logger.info(f"Loading checkpoint: {current_ckpt_path}")
            # logger.info(f"Freeze loaded weights: {freeze_next_stage}") # Removed freeze logic log

            # Call the train function for the current stage
            next_ckpt_path = train(
                stage=stage,
                ckpt_path=current_ckpt_path,
                config_path=config_path,
                # freeze_weights=freeze_next_stage # Removed freeze_weights arg
            )

            # Check if the stage completed successfully and produced a checkpoint
            if next_ckpt_path is None:
                logger.error(f"Stage {stage} failed or did not produce a checkpoint. Stopping sequential training.")
                break # Exit the loop if a stage fails

            # Prepare for the next stage
            current_ckpt_path = next_ckpt_path
            # freeze_next_stage = True # Removed freeze logic

            # Check if this was the last stage
            if stage == 3:
                logger.info("--- Sequential Training (Stages 1-3) Completed Successfully! ---")
                logger.info(f"Final checkpoint saved at: {current_ckpt_path}")

        # This else block executes if the loop completes without break
        # else:
        #     print("\n--- Sequential Training (Stages 1-3) Completed Successfully! ---")
        #     # The final checkpoint path is already printed in the last iteration