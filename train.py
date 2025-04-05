# train.py
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Import custom modules
from data.dataset import DataModule # Assuming DataModule is in data/dataset.py
from model import SVSGAN            # Import the new SVSGAN model

def train(config_path='config/model.yaml'):
    """
    Main training function.

    Args:
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

    # --- Setup ---
    # Ensure log directories exist

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
    print("Initializing SVSGAN model...")
    model = SVSGAN(config)
    print("Model initialized successfully.")

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
        dirpath=ckpt_dir,
        filename='svs-gan-{epoch:02d}-{val_loss:.2f}', # Updated filename for GAN model
        save_top_k=1,          # Save only the best model
        monitor='val_loss',    # Monitor validation loss
        mode='min',            # Mode should be 'min' for loss
        save_last=True         # Optionally save the last checkpoint as well
    )
    print("Logger and callbacks configured.")

    # --- 5. Initialize Trainer ---
    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config['train']['epochs'],
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
        trainer.fit(model, datamodule=data_module)
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Consider more specific error handling if needed

if __name__ == "__main__":
    # Set random seed for reproducibility (optional but recommended)
    pl.seed_everything(42, workers=True)

    train() # Use default config path 'config/model.yaml'