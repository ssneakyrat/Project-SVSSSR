import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np
import math
import os


class H5FileManager:
    """
    Singleton to manage H5 file connections across the application
    """
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


class SVSDataset(Dataset):
    """
    Dataset for Singing Voice Synthesis model training
    Loads mel spectrograms, f0 contours, phone labels, durations, and MIDI notes
    """
    def __init__(self, h5_path, data_key='mel_spectrograms', transform=None, lazy_load=True):
        self.h5_path = h5_path
        self.data_key = data_key
        self.transform = transform
        self.lazy_load = lazy_load
        
        # Open H5 file
        if lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(h5_path)
        else:
            h5_file = h5py.File(h5_path, 'r')
            
        # Verify that data exists
        if data_key in h5_file:
            self.length = len(h5_file[data_key])
            self.data_shape = h5_file[data_key].shape[1:]
        else:
            raise KeyError(f"Data key '{data_key}' not found in {h5_path}")
        
        # Store metadata
        self.has_f0 = 'f0' in h5_file
        self.has_phone = 'phone' in h5_file
        self.has_duration = 'duration' in h5_file
        self.has_midi = 'midi' in h5_file
        
        # Load all data at once if not using lazy loading
        if not lazy_load:
            self.mel_specs = h5_file[data_key][:]
            
            if self.has_f0:
                self.f0 = h5_file['f0'][:]
            
            if self.has_phone:
                self.phone = h5_file['phone'][:]
            
            if self.has_duration:
                self.duration = h5_file['duration'][:]
            
            if self.has_midi:
                self.midi = h5_file['midi'][:]
                
            h5_file.close()
        
        # Store audio config attributes if available
        if lazy_load:
            if data_key in h5_file and hasattr(h5_file[data_key], 'attrs'):
                self.attrs = dict(h5_file[data_key].attrs)
            else:
                self.attrs = {}
        else:
            self.attrs = {}
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load data
        if self.lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(self.h5_path)
            
            # Load mel spectrogram
            mel_spec = h5_file[self.data_key][idx]
            mel_spec = torch.from_numpy(mel_spec).float()
            
            # Prepare result dictionary
            result = {'mel': mel_spec.unsqueeze(0)}  # Add channel dimension
            
            # Load additional features if available
            if self.has_f0:
                f0 = h5_file['f0'][idx]
                result['f0'] = torch.from_numpy(f0).float()
            else:
                # Generate placeholder f0 contour
                result['f0'] = torch.zeros(mel_spec.shape[1]).float()
            
            if self.has_phone:
                phone = h5_file['phone'][idx]
                result['phone'] = torch.from_numpy(phone).long()
            else:
                # Generate placeholder phone labels
                result['phone'] = torch.zeros(mel_spec.shape[1], dtype=torch.long)
            
            if self.has_duration:
                duration = h5_file['duration'][idx]
                result['duration'] = torch.from_numpy(duration).float()
            else:
                # Generate placeholder durations
                result['duration'] = torch.ones(mel_spec.shape[1]).float()
            
            if self.has_midi:
                midi = h5_file['midi'][idx]
                result['midi'] = torch.from_numpy(midi).long()
            else:
                # Generate placeholder MIDI notes (C4 = 60)
                result['midi'] = torch.full((mel_spec.shape[1],), 60, dtype=torch.long)
        else:
            # Get from preloaded data
            mel_spec = torch.from_numpy(self.mel_specs[idx]).float()
            
            # Prepare result dictionary
            result = {'mel': mel_spec.unsqueeze(0)}  # Add channel dimension
            
            # Add additional features if available
            if self.has_f0:
                result['f0'] = torch.from_numpy(self.f0[idx]).float()
            else:
                result['f0'] = torch.zeros(mel_spec.shape[1]).float()
                
            if self.has_phone:
                result['phone'] = torch.from_numpy(self.phone[idx]).long()
            else:
                result['phone'] = torch.zeros(mel_spec.shape[1], dtype=torch.long)
                
            if self.has_duration:
                result['duration'] = torch.from_numpy(self.duration[idx]).float()
            else:
                result['duration'] = torch.ones(mel_spec.shape[1]).float()
                
            if self.has_midi:
                result['midi'] = torch.from_numpy(self.midi[idx]).long()
            else:
                result['midi'] = torch.full((mel_spec.shape[1],), 60, dtype=torch.long)
        
        # Apply transforms if provided
        if self.transform:
            result = self.transform(result)
            
        return result


class SVSDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for SVS training
    """
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
        self.sample_percentage = config['data'].get('sample_percentage', None)
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            # Create dataset
            self.dataset = SVSDataset(
                h5_path=self.h5_path,
                data_key=self.data_key,
                lazy_load=self.lazy_load
            )
            
            # Apply limits if specified
            full_dataset_size = len(self.dataset)
            subset_size = full_dataset_size
            
            if self.max_samples and self.max_samples > 0:
                subset_size = min(self.max_samples, full_dataset_size)
            elif self.sample_percentage and 0.0 < self.sample_percentage <= 1.0:
                subset_size = int(full_dataset_size * self.sample_percentage)
            
            # Create subset if needed
            if subset_size < full_dataset_size:
                generator = torch.Generator().manual_seed(42)
                indices = torch.randperm(full_dataset_size, generator=generator)[:subset_size].tolist()
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
            
            # Split into train and validation
            dataset_size = len(self.dataset)
            val_size = int(dataset_size * self.validation_split)
            train_size = dataset_size - val_size
            
            generator = torch.Generator().manual_seed(42)
            
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, 
                [train_size, val_size],
                generator=generator
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()


def collate_variable_length(batch):
    """
    Custom collate function for variable length inputs
    """
    # Find max lengths in the batch
    max_mel_length = max([item['mel'].shape[2] for item in batch])
    max_seq_length = max([item['f0'].shape[0] for item in batch])
    
    batch_size = len(batch)
    
    # Create tensors to hold the batch
    batched_mel = torch.zeros((batch_size, 1, batch[0]['mel'].shape[1], max_mel_length))
    batched_f0 = torch.zeros((batch_size, max_seq_length))
    batched_phone = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
    batched_duration = torch.zeros((batch_size, max_seq_length))
    batched_midi = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
    
    # Create masks to track actual sequence lengths
    mel_mask = torch.zeros((batch_size, max_mel_length), dtype=torch.bool)
    seq_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.bool)
    
    # Fill in the batch tensors
    for i, item in enumerate(batch):
        # Get actual lengths
        mel_len = item['mel'].shape[2]
        seq_len = item['f0'].shape[0]
        
        # Place the data in the batch tensors
        batched_mel[i, :, :, :mel_len] = item['mel']
        batched_f0[i, :seq_len] = item['f0']
        batched_phone[i, :seq_len] = item['phone']
        batched_duration[i, :seq_len] = item['duration']
        batched_midi[i, :seq_len] = item['midi']
        
        # Mark actual data positions in the masks
        mel_mask[i, :mel_len] = 1
        seq_mask[i, :seq_len] = 1
    
    return {
        'mel': batched_mel,
        'f0': batched_f0,
        'phone': batched_phone,
        'duration': batched_duration,
        'midi': batched_midi,
        'mel_mask': mel_mask,
        'seq_mask': seq_mask
    }