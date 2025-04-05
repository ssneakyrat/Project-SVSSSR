import torch
import numpy as np
import h5py
from data.dataset import H5Dataset, H5FileManager, DataModule

class SVSDataset(H5Dataset):
    """
    Dataset for Singing Voice Synthesis that provides phone, f0, duration, midi inputs
    and ground truth mel spectrograms.
    """
    def __init__(self, h5_path, data_key, phone_key='phone_ids', f0_key='f0', 
                 duration_key='durations', midi_key='midi_ids', 
                 transform=None, lazy_load=True, variable_length=False):
        """
        Args:
            h5_path: Path to the H5 file
            data_key: Key for the mel spectrogram data
            phone_key: Key for the phone ID data
            f0_key: Key for the F0 contour data
            duration_key: Key for the duration data
            midi_key: Key for the MIDI ID data
            transform: Optional transform to apply to the data
            lazy_load: Whether to load data lazily
            variable_length: Whether to support variable length data
        """
        super().__init__(h5_path, data_key, transform, lazy_load, variable_length)
        self.phone_key = phone_key
        self.f0_key = f0_key
        self.duration_key = duration_key
        self.midi_key = midi_key
    
    def __getitem__(self, idx):
        """
        Get a data sample by index.
        
        Returns:
            tuple: (phone_ids, f0, durations, midi_ids, mel_spectrogram)
        """
        if self.lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(self.h5_path)
            
            # Load ground truth mel spectrogram
            mel_data = h5_file[self.data_key][idx]
            mel_data = torch.from_numpy(mel_data).float()
            
            # Load input features (actual implementation would load all these)
            # For now, we'll create placeholder data if the keys don't exist
            try:
                phone_ids = h5_file[self.phone_key][idx]
                phone_ids = torch.from_numpy(phone_ids).long()
            except (KeyError, IndexError):
                # Create placeholder phone IDs with correct time dimension
                phone_ids = torch.zeros(mel_data.shape[1], dtype=torch.long)
            
            try:
                f0_data = h5_file[self.f0_key][idx]
                f0_data = torch.from_numpy(f0_data).float().unsqueeze(-1)  # [time, 1]
            except (KeyError, IndexError):
                # Create placeholder F0 data with correct time dimension
                f0_data = torch.zeros((mel_data.shape[1], 1), dtype=torch.float)
            
            try:
                duration_data = h5_file[self.duration_key][idx]
                duration_data = torch.from_numpy(duration_data).float().unsqueeze(-1)  # [time, 1]
            except (KeyError, IndexError):
                # Create placeholder duration data with correct time dimension
                duration_data = torch.zeros((mel_data.shape[1], 1), dtype=torch.float)
            
            try:
                midi_ids = h5_file[self.midi_key][idx]
                midi_ids = torch.from_numpy(midi_ids).long()
            except (KeyError, IndexError):
                # Create placeholder MIDI IDs with correct time dimension
                midi_ids = torch.zeros(mel_data.shape[1], dtype=torch.long)
        else:
            # For non-lazy loading, we would load from self.data
            # This is a placeholder implementation
            mel_data = torch.from_numpy(self.data[idx]).float()
            # Create placeholder data with correct dimensions
            phone_ids = torch.zeros(mel_data.shape[1], dtype=torch.long)
            f0_data = torch.zeros((mel_data.shape[1], 1), dtype=torch.float)
            duration_data = torch.zeros((mel_data.shape[1], 1), dtype=torch.float)
            midi_ids = torch.zeros(mel_data.shape[1], dtype=torch.long)
        
        # Ensure mel_data is in the correct format [freq_bins, time_frames]
        if mel_data.dim() == 2:
            # Check if the dimensions are flipped (time_frames, freq_bins)
            if mel_data.shape[0] > mel_data.shape[1]:  # If time > freq, transpose
                mel_data = mel_data.transpose(0, 1)
            
            # Add channel dimension [1, freq_bins, time_frames]
            mel_data = mel_data.unsqueeze(0)
        
        if self.transform:
            mel_data = self.transform(mel_data)
        
        return phone_ids, f0_data, duration_data, midi_ids, mel_data


def collate_variable_length_svs(batch):
    """
    Custom collate function for variable length SVS data
    
    Args:
        batch: List of tuples (phone_ids, f0, durations, midi_ids, mel_spectrogram)
        
    Returns:
        tuple: Batched tensors with masks
    """
    # Extract components
    phone_ids, f0_data, duration_data, midi_ids, mel_specs = zip(*batch)
    
    # Find max lengths
    max_seq_len = max([p.shape[0] for p in phone_ids])
    max_mel_len = max([m.shape[2] for m in mel_specs])
    
    batch_size = len(phone_ids)
    mel_channels = mel_specs[0].shape[0]
    mel_bins = mel_specs[0].shape[1]
    
    # Create tensors to hold the batch data
    batched_phone_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    batched_f0 = torch.zeros((batch_size, max_seq_len, 1), dtype=torch.float)
    batched_durations = torch.zeros((batch_size, max_seq_len, 1), dtype=torch.float)
    batched_midi_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    batched_mel_specs = torch.zeros((batch_size, mel_channels, mel_bins, max_mel_len), dtype=torch.float)
    
    # Create masks
    seq_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    mel_mask = torch.zeros((batch_size, max_mel_len), dtype=torch.bool)
    
    # Fill tensors
    for i, (p, f, d, m, mel) in enumerate(zip(phone_ids, f0_data, duration_data, midi_ids, mel_specs)):
        # Get lengths
        seq_len = p.shape[0]
        mel_len = mel.shape[2]
        
        # Fill data
        batched_phone_ids[i, :seq_len] = p
        batched_f0[i, :seq_len, :] = f
        batched_durations[i, :seq_len, :] = d
        batched_midi_ids[i, :seq_len] = m
        batched_mel_specs[i, :, :, :mel_len] = mel
        
        # Fill masks
        seq_mask[i, :seq_len] = 1
        mel_mask[i, :mel_len] = 1
    
    return (
        batched_phone_ids, 
        batched_f0, 
        batched_durations, 
        batched_midi_ids, 
        batched_mel_specs, 
        seq_mask, 
        mel_mask
    )


class SVSDataModule(DataModule):
    """
    Data module for SVS training that handles loading of all necessary inputs
    """
    def __init__(self, config):
        super().__init__(config)
        self.phone_key = config.get('data', {}).get('phone_key', 'phone_ids')
        self.f0_key = config.get('data', {}).get('f0_key', 'f0')
        self.duration_key = config.get('data', {}).get('duration_key', 'durations')
        self.midi_key = config.get('data', {}).get('midi_key', 'midi_ids')
    
    def setup(self, stage=None):
        if self.dataset is None:
            # Use SVSDataset instead of regular dataset
            if self.variable_length:
                self.dataset = SVSDataset(
                    h5_path=self.h5_path,
                    data_key=self.data_key,
                    phone_key=self.phone_key,
                    f0_key=self.f0_key,
                    duration_key=self.duration_key,
                    midi_key=self.midi_key,
                    lazy_load=self.lazy_load,
                    variable_length=True
                )
            else:
                self.dataset = SVSDataset(
                    h5_path=self.h5_path,
                    data_key=self.data_key,
                    phone_key=self.phone_key,
                    f0_key=self.f0_key,
                    duration_key=self.duration_key,
                    midi_key=self.midi_key,
                    lazy_load=self.lazy_load,
                    variable_length=False
                )
            
            # Subset the dataset if needed
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
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, 
                [train_size, val_size],
                generator=generator
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True,
            collate_fn=collate_variable_length_svs if self.variable_length else None
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=collate_variable_length_svs if self.variable_length else None
        )