import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np
import torch.multiprocessing as mp

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
        pid = os.getpid()
        file_key = f"{pid}_{file_path}"
        if file_key not in self.h5_files:
            self.h5_files[file_key] = h5py.File(file_path, 'r')
        return self.h5_files[file_key]
    
    def close_all(self):
        for file in self.h5_files.values():
            file.close()
        self.h5_files = {}
        
    def __getstate__(self):
        return {'initialized': True}
    
    def __setstate__(self, state):
        self.h5_files = {}

class SVSDataset(Dataset):
    def __init__(self, h5_path, data_key='mel_spectrograms', lazy_load=True):
        self.h5_path = h5_path
        self.data_key = data_key
        self.lazy_load = lazy_load
        
        with h5py.File(h5_path, 'r') as h5_file:
            if data_key in h5_file:
                self.length = len(h5_file[data_key])
                self.data_shape = h5_file[data_key].shape[1:]
                self.variable_length = h5_file[data_key].attrs.get('variable_length', False)
            else:
                raise KeyError(f"Data key '{data_key}' not found in {h5_path}")
            
            if not lazy_load:
                self.data = h5_file[data_key][:]
                if 'lengths' in h5_file:
                    self.lengths = h5_file['lengths'][:]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if not self.lazy_load and hasattr(self, 'data'):
            mel_spec = torch.from_numpy(self.data[idx]).float()
            length = self.lengths[idx] if hasattr(self, 'lengths') else mel_spec.shape[1]
            
            with h5py.File(self.h5_path, 'r') as h5_file:
                f0 = torch.from_numpy(h5_file['f0'][idx] if 'f0' in h5_file else np.zeros(mel_spec.shape[1])).float()
                phone_label = torch.from_numpy(h5_file['phone_label'][idx] if 'phone_label' in h5_file else np.zeros(mel_spec.shape[1], dtype=np.int64)).long()
                phone_duration = torch.from_numpy(h5_file['phone_duration'][idx] if 'phone_duration' in h5_file else np.ones(1, dtype=np.float32)).float()
                midi_label = torch.from_numpy(h5_file['midi_label'][idx] if 'midi_label' in h5_file else np.zeros(mel_spec.shape[1], dtype=np.int64) + 60).long()
        else:
            with h5py.File(self.h5_path, 'r') as h5_file:
                mel_spec = torch.from_numpy(h5_file[self.data_key][idx]).float()
                
                if self.variable_length and 'lengths' in h5_file:
                    length = h5_file['lengths'][idx]
                else:
                    length = mel_spec.shape[1]
                
                f0 = torch.from_numpy(h5_file['f0'][idx] if 'f0' in h5_file else np.zeros(mel_spec.shape[1])).float()
                phone_label = torch.from_numpy(h5_file['phone_label'][idx] if 'phone_label' in h5_file else np.zeros(mel_spec.shape[1], dtype=np.int64)).long()
                phone_duration = torch.from_numpy(h5_file['phone_duration'][idx] if 'phone_duration' in h5_file else np.ones(1, dtype=np.float32)).float()
                midi_label = torch.from_numpy(h5_file['midi_label'][idx] if 'midi_label' in h5_file else np.zeros(mel_spec.shape[1], dtype=np.int64) + 60).long()
        
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
    
    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        
        max_mel_length = max([item['mel_spec'].shape[1] for item in batch])
        mel_bins = batch[0]['mel_spec'].shape[0]
        
        mel_specs = torch.zeros(batch_size, mel_bins, max_mel_length)
        f0s = torch.zeros(batch_size, max_mel_length)
        phone_labels = torch.zeros(batch_size, max_mel_length, dtype=torch.long)
        midi_labels = torch.zeros(batch_size, max_mel_length, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, item in enumerate(batch):
            item_length = item['length']
            lengths[i] = item_length
            
            mel_spec = item['mel_spec']
            mel_specs[i, :, :item_length] = mel_spec[:, :item_length]
            
            f0 = item['f0']
            f0_length = min(f0.shape[0], max_mel_length)
            f0s[i, :f0_length] = f0[:f0_length]
            
            phone_label = item['phone_label']
            phone_length = min(phone_label.shape[0], max_mel_length)
            phone_labels[i, :phone_length] = phone_label[:phone_length]
            
            midi_label = item['midi_label']
            midi_length = min(midi_label.shape[0], max_mel_length)
            midi_labels[i, :midi_length] = midi_label[:midi_length]
        
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
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()

def worker_init_fn(worker_id):
    import numpy as np
    import random
    import torch
    
    base_seed = 42
    worker_seed = base_seed + worker_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        
        if hasattr(dataset, '_h5_file') and dataset._h5_file is not None:
            try:
                dataset._h5_file.close()
            except:
                pass
            finally:
                dataset._h5_file = None

def check_dataset(h5_path, variable_length=True):
    if not os.path.exists(h5_path):
        print(f"Error: Dataset file {h5_path} not found!")
        return False
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'mel_spectrograms' not in f:
                print(f"Error: 'mel_spectrograms' not found in {h5_path}")
                return False
            
            if variable_length and 'lengths' not in f:
                print(f"Warning: Variable length mode enabled but 'lengths' not found in {h5_path}")
                print("The dataset might not be prepared for variable length. Consider reprocessing.")
                
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

def fix_multiprocessing():
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn' for better compatibility")
    except RuntimeError:
        print("Could not set multiprocessing start method to 'spawn'")
    
    torch.multiprocessing.set_sharing_strategy('file_system')