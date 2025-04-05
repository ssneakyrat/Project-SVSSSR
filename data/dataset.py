import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np
import math

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

class H5Dataset(Dataset):
    def __init__(self, h5_path, data_key, transform=None, lazy_load=True, variable_length=False):
        self.h5_path = h5_path
        self.data_key = data_key
        self.transform = transform
        self.lazy_load = lazy_load
        self.variable_length = variable_length
        
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
            
        if not lazy_load:
            self.data = h5_file[data_key][:]
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
        if self.lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(self.h5_path)
            data = h5_file[self.data_key][idx]
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(self.data[idx]).float()
                
        if self.transform:
            data = self.transform(data)
            
        return data

class MelSpectrogramDataset(H5Dataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # Make sure data is in the correct format [freq_bins, time_frames]
        if data.dim() == 2:
            # Check if the dimensions are flipped (time_frames, freq_bins)
            if data.shape[0] > data.shape[1]:  # If time > freq, transpose
                data = data.transpose(0, 1)
                
            # Add channel dimension [1, freq_bins, time_frames]
            data = data.unsqueeze(0)
        
        return data

class VariableLengthMelDataset(H5Dataset):
    """Dataset that supports variable length mel spectrograms with max length constraint"""
    def __init__(self, h5_path, data_key, max_frames=None, transform=None, lazy_load=True):
        super().__init__(h5_path, data_key, transform, lazy_load, variable_length=True)
        self.max_frames = max_frames
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # If data is already 3D, make sure it's in the correct format [channels, freq, time]
        if data.dim() == 3:
            # Check if the dimensions are flipped [channels, time, freq]
            if data.shape[1] > data.shape[2]:  # If time > freq, transpose
                data = data.transpose(1, 2)
        elif data.dim() == 2:
            # For 2D tensors [freq, time] or [time, freq]
            if data.shape[0] > data.shape[1]:  # If time > freq, transpose
                data = data.transpose(0, 1)
            # Add channel dimension
            data = data.unsqueeze(0)
            
        # Check if we need to limit the time frames
        if self.max_frames is not None and data.shape[2] > self.max_frames:
            # Trim to max_frames (in dim 2, which is time dimension)
            data = data[:, :, :self.max_frames]
        
        return data

def collate_variable_length(batch):
    """Custom collate function for variable length mel spectrograms"""
    # Find max length in the batch (time dimension = dim 2)
    max_length = max([item.shape[2] for item in batch])
    batch_size = len(batch)
    channels = batch[0].shape[0]
    height = batch[0].shape[1]  # freq bins
    
    # Create tensor to hold the batch - [batch, channels, height, max_length]
    batched_data = torch.zeros((batch_size, channels, height, max_length))
    
    # Create a mask to track actual sequence lengths (1 for data, 0 for padding)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    
    # Fill in the batch tensor
    for i, item in enumerate(batch):
        # Get sequence length (time dimension)
        seq_len = item.shape[2]
        # Place the data in the batch tensor
        batched_data[i, :, :, :seq_len] = item
        # Mark actual data positions in the mask
        mask[i, :seq_len] = 1
    
    return batched_data, mask

class DataModule(pl.LightningDataModule):
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
        
        # Handle variable length inputs
        self.variable_length = config['data'].get('variable_length', False)
        
        # Calculate max time frames for 10 seconds of audio
        if 'max_audio_length' in config['audio']:
            self.max_audio_length = config['audio']['max_audio_length']
            sample_rate = config['audio']['sample_rate']
            hop_length = config['audio']['hop_length']
            self.max_frames = math.ceil(self.max_audio_length * sample_rate / hop_length)
        else:
            self.max_frames = config['model'].get('time_frames', 128)
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            # Choose appropriate dataset class based on variable_length flag
            if self.variable_length:
                self.dataset = VariableLengthMelDataset(
                    h5_path=self.h5_path,
                    data_key=self.data_key,
                    max_frames=self.max_frames,
                    lazy_load=self.lazy_load
                )
            else:
                self.dataset = MelSpectrogramDataset(
                    h5_path=self.h5_path,
                    data_key=self.data_key,
                    lazy_load=self.lazy_load
                )
            
            full_dataset_size = len(self.dataset)
            subset_size = full_dataset_size
            
            if self.max_samples and self.max_samples > 0:
                subset_size = min(self.max_samples, full_dataset_size)
            elif self.sample_percentage and 0.0 < self.sample_percentage <= 1.0:
                subset_size = int(full_dataset_size * self.sample_percentage)
            
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
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True,
            collate_fn=collate_variable_length if self.variable_length else None
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=collate_variable_length if self.variable_length else None
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()