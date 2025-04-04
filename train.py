import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
import sys

# Add current directory to path if needed
sys.path.append('.')

from utils.utils import load_config
from models.progressive_svs import ProgressiveSVS

# Import directly from dataset.py to avoid module structure issues
class H5FileManager:
    _instance = None
    
    @staticmethod
    def get_instance():
        if H5FileManager._instance is None:
            H5FileManager._instance = H5FileManager()
        return H5FileManager._instance
    
    def __init__(self):
        self.h5_files = {}
    
    def get_file(self, file_path):
        if file_path not in self.h5_files:
            self.h5_files[file_path] = h5py.File(file_path, 'r')
        return self.h5_files[file_path]
    
    def close_all(self):
        for file in self.h5_files.values():
            file.close()
        self.h5_files = {}

# Dataset and DataModule classes (copied from updated dataset.py)
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np

class SVSDataset(Dataset):
    def __init__(self, h5_path, data_key='mel_spectrograms', lazy_load=True):
        self.h5_path = h5_path
        self.data_key = data_key
        self.lazy_load = lazy_load
        
        if lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(h5_path)
        else:
            h5_file = h5py.File(h5_path, 'r')
            
        if data_key in h5_file:
            self.length = len(h5_file[data_key])
            self.data_shape = h5_file[data_key].shape[1:]
        else:
            raise KeyError(f"Data key '{data_key}' not found in {h5_path}")
            
        self.variable_length = h5_file[data_key].attrs.get('variable_length', False)
        
        if not lazy_load:
            self.data = h5_file[data_key][:]
            if 'lengths' in h5_file:
                self.lengths = h5_file['lengths'][:]
            h5_file.close()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        h5_manager = H5FileManager.get_instance()
        h5_file = h5_manager.get_file(self.h5_path)
        
        # Get mel spectrogram (target)
        mel_spec = h5_file[self.data_key][idx]
        mel_spec = torch.from_numpy(mel_spec).float()
        
        # Get actual length if variable length mode
        if self.variable_length and 'lengths' in h5_file:
            length = h5_file['lengths'][idx]
        else:
            length = mel_spec.shape[1]  # Assume full length if not specified
        
        # Get F0 contour
        f0 = None
        if 'f0' in h5_file:
            f0 = h5_file['f0'][idx]
            f0 = torch.from_numpy(f0).float()
        else:
            f0 = torch.zeros(mel_spec.shape[1])
        
        # Get phone labels
        phone_label = None
        if 'phone_label' in h5_file:
            phone_label = h5_file['phone_label'][idx]
            phone_label = torch.from_numpy(phone_label).long()
        else:
            phone_label = torch.zeros(mel_spec.shape[1], dtype=torch.long)
        
        # Get phone durations - optional
        phone_duration = None
        if 'phone_duration' in h5_file:
            phone_duration = h5_file['phone_duration'][idx]
            phone_duration = torch.from_numpy(phone_duration).float()
        else:
            phone_duration = torch.ones(1, dtype=torch.float)  # Default duration
        
        # Get MIDI labels - optional
        midi_label = None
        if 'midi_label' in h5_file:
            midi_label = h5_file['midi_label'][idx]
            midi_label = torch.from_numpy(midi_label).long()
        else:
            midi_label = torch.zeros(mel_spec.shape[1], dtype=torch.long) + 60  # Default to middle C
        
        return {
            'mel_spec': mel_spec,
            'f0': f0,
            'phone_label': phone_label,
            'phone_duration': phone_duration,
            'midi_label': midi_label,
            'length': length
        }

class SVSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)
        
        self.h5_path = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
        self.data_key = config['data'].get('data_key', 'mel_spectrograms')
        self.lazy_load = config['data'].get('lazy_load', True)
        self.max_samples = config['data'].get('max_samples', None)
        
    def setup(self, stage=None):
        self.dataset = SVSDataset(
            h5_path=self.h5_path,
            data_key=self.data_key,
            lazy_load=self.lazy_load
        )
        
        full_dataset_size = len(self.dataset)
        subset_size = full_dataset_size
        
        if self.max_samples and self.max_samples > 0:
            subset_size = min(self.max_samples, full_dataset_size)
            
        if subset_size < full_dataset_size:
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(full_dataset_size, generator=generator)[:subset_size].tolist()
            self.dataset = torch.utils.data.Subset(self.dataset, indices)
        
        dataset_size = len(self.dataset)
        val_size = int(dataset_size * self.validation_split)
        train_size = dataset_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, 
            [train_size, val_size],
            generator=generator
        )
    
    def collate_fn(self, batch):
        # Handle variable length sequences
        batch_size = len(batch)
        
        # Find maximum length in this batch
        max_mel_length = max([item['mel_spec'].shape[1] for item in batch])
        mel_bins = batch[0]['mel_spec'].shape[0]
        
        # Create padded tensors
        mel_specs = torch.zeros(batch_size, mel_bins, max_mel_length)
        f0s = torch.zeros(batch_size, max_mel_length)
        phone_labels = torch.zeros(batch_size, max_mel_length, dtype=torch.long)
        midi_labels = torch.zeros(batch_size, max_mel_length, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill with data
        for i, item in enumerate(batch):
            item_length = item['length']
            lengths[i] = item_length
            
            # Handle mel spectrogram
            mel_spec = item['mel_spec']
            mel_specs[i, :, :item_length] = mel_spec[:, :item_length]
            
            # Handle f0
            f0 = item['f0']
            f0_length = min(f0.shape[0], max_mel_length)
            f0s[i, :f0_length] = f0[:f0_length]
            
            # Handle phone labels
            phone_label = item['phone_label']
            phone_length = min(phone_label.shape[0], max_mel_length)
            phone_labels[i, :phone_length] = phone_label[:phone_length]
            
            # Handle midi labels
            midi_label = item['midi_label']
            midi_length = min(midi_label.shape[0], max_mel_length)
            midi_labels[i, :midi_length] = midi_label[:midi_length]
        
        # Collect phone durations (different format)
        phone_durations = [item['phone_duration'] for item in batch]
        
        return {
            'mel_spec': mel_specs,
            'f0': f0s,
            'phone_label': phone_labels,
            'phone_duration': phone_durations,
            'midi_label': midi_labels,
            'length': lengths
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()

def check_dataset(h5_path, variable_length=True):
    """Verify that the dataset exists and has the expected format."""
    if not os.path.exists(h5_path):
        print(f"Error: Dataset file {h5_path} not found!")
        return False
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if the file has mel_spectrograms dataset
            if 'mel_spectrograms' not in f:
                print(f"Error: 'mel_spectrograms' not found in {h5_path}")
                return False
            
            # Check if the file has lengths dataset (for variable length)
            if variable_length and 'lengths' not in f:
                print(f"Warning: Variable length mode enabled but 'lengths' not found in {h5_path}")
                print("The dataset might not be prepared for variable length. Consider reprocessing.")
                
            # Check basic structure
            if 'phone_map' not in f:
                print(f"Warning: 'phone_map' not found in {h5_path}")
                
            if 'phone_label' not in f:
                print(f"Warning: 'phone_label' not found in {h5_path}")
                
            if 'f0' not in f:
                print(f"Warning: 'f0' not found in {h5_path}")
    
    except Exception as e:
        print(f"Error checking dataset: {e}")
        return False
        
    return True

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
    
    # Override current stage if specified
    if args.stage:
        config['model']['current_stage'] = args.stage
    
    # Override variable_length if specified
    if args.variable_length:
        config['data']['variable_length'] = True
    
    # Override batch size if specified
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    
    # Override max epochs if specified
    if args.max_epochs:
        config['train']['num_epochs'] = args.max_epochs
    
    # Get current stage
    current_stage = config['model']['current_stage']
    stage_epochs = config['model'].get('stage_epochs', [30, 30, 40])
    
    # Set epochs based on stage (if not manually overridden)
    if not args.max_epochs and current_stage <= len(stage_epochs):
        config['train']['num_epochs'] = stage_epochs[current_stage-1]
    
    # Check if dataset exists and has expected format
    h5_path = os.path.join(config['data']['bin_dir'], config['data']['bin_file'])
    variable_length = config['data'].get('variable_length', True)
    
    if not check_dataset(h5_path, variable_length):
        print(f"Dataset check failed. Please ensure {h5_path} exists and has the required format.")
        print("You may need to run preprocess.py first.")
        return
    
    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Create logger with proper versioning
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=f'stage{current_stage}',
        version=None  # Auto-increment version
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
        # Initialize model and data module
        model = ProgressiveSVS(config)
        data_module = SVSDataModule(config)
        
        # Configure trainer
        trainer_kwargs = {
            'max_epochs': config['train']['num_epochs'],
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': 10,
            'accelerator': 'auto',
            'devices': 'auto',
        }
        
        # Add gradient clipping for stability
        if 'gradient_clip_val' in config['train']:
            trainer_kwargs['gradient_clip_val'] = config['train']['gradient_clip_val']
        
        trainer = pl.Trainer(**trainer_kwargs)
        
        # Train the model
        trainer.fit(model, data_module, ckpt_path=args.resume)
        
        print(f"Training completed for stage {current_stage}.")
        
        # Suggest moving to next stage if applicable
        if current_stage < 3:
            next_stage = current_stage + 1
            checkpoint_path = os.path.join(logger.log_dir, "checkpoints/last.ckpt")
            print(f"\nTo continue training with the next stage, run:")
            print(f"python train.py --stage {next_stage} --resume {checkpoint_path}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Clean up H5 files
        H5FileManager.get_instance().close_all()
        print("H5 file resources released.")

if __name__ == "__main__":
    main()