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
    """
    Dataset to load multiple data types (e.g., mel, phonemes, f0, durations)
    from different keys within a single HDF5 file.
    Assumes all datasets referenced by keys have the same number of samples.
    """
    def __init__(self, h5_path, data_keys, transform=None, lazy_load=True):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            data_keys (dict): Dictionary mapping descriptive names to HDF5 dataset keys
                              (e.g., {'mel': 'mel_spectrograms', 'phoneme': 'phonemes'}).
            transform (callable, optional): Optional transform to be applied on a sample.
                                            (Note: May need key-specific transforms later).
            lazy_load (bool): Whether to load data on demand or all at once.
        """
        self.h5_path = h5_path
        self.data_keys = data_keys
        self.transform = transform # TODO: Implement key-specific transforms if needed
        self.lazy_load = lazy_load
        self.h5_file_handle = None # For non-lazy loading

        if not self.data_keys:
            raise ValueError("data_keys dictionary cannot be empty.")

        self.sample_keys = [] # List to store HDF5 group keys (representing samples)
        self.length = 0

        h5_file = None
        try:
            if self.lazy_load:
                h5_manager = H5FileManager.get_instance()
                h5_file = h5_manager.get_file(h5_path)
            else:
                # Keep file open if not lazy loading
                self.h5_file_handle = h5py.File(h5_path, 'r')
                h5_file = self.h5_file_handle

            # Get sample keys (group names) and determine length
            self.sample_keys = list(h5_file.keys())
            if not self.sample_keys:
                raise ValueError(f"No sample groups found in HDF5 file: {h5_path}")
            self.length = len(self.sample_keys)

            # Check if expected data keys exist within the first sample group (sanity check)
            if self.sample_keys: # Ensure there's at least one group
                first_group_name = self.sample_keys[0]
                try:
                    first_group = h5_file[first_group_name]
                    for h5_key in self.data_keys.values():
                        if h5_key not in first_group:
                            # Warning or error depending on strictness
                            print(f"Warning: Data key '{h5_key}' not found in the first sample group '{first_group_name}' of {h5_path}. Assuming it exists in others.")
                            # raise KeyError(f"Data key '{h5_key}' not found in the first sample group '{first_group_name}' of {h5_path}")
                except KeyError:
                     print(f"Warning: Could not access first group '{first_group_name}' for key check in {h5_path}.")
            # Pre-load all data if not lazy loading
            if not self.lazy_load:
                print(f"Pre-loading data from {self.h5_path}...")
                self.data = {} # Store pre-loaded data indexed by group_name
                for group_name in self.sample_keys:
                    self.data[group_name] = {}
                    group = h5_file[group_name]
                    for name, h5_key in self.data_keys.items():
                        if h5_key in group:
                            self.data[group_name][name] = group[h5_key][()] # Load data into memory
                        else:
                            self.data[group_name][name] = None # Or handle differently
                            print(f"Warning: Key '{h5_key}' not found in group '{group_name}' during pre-loading.")
                print("Pre-loading complete.")
                # Keep handle open if not lazy loading, as __getitem__ might still use it if pre-loading fails for a key
        except Exception as e:
            # Ensure file handle is closed on error during init if not lazy
            if self.h5_file_handle:
                self.h5_file_handle.close()
            raise e
        # Note: For lazy loading, H5FileManager keeps the file open until teardown.

    def __len__(self):
        return self.length

    
    def __getitem__(self, idx):
        sample = {}
        h5_file = None # Define h5_file scope

        try:
            if self.lazy_load:
                h5_manager = H5FileManager.get_instance()
                h5_file = h5_manager.get_file(self.h5_path)
            else:
                # Use pre-loaded data or the persistent handle
                if hasattr(self, 'data'): # Data was pre-loaded
                     group_name = self.sample_keys[idx]
                     if group_name not in self.data:
                          # This might happen if pre-loading failed for this group earlier
                          print(f"Warning: Group name '{group_name}' (index {idx}) not found in pre-loaded data. Attempting direct load.")
                          # Fall back to using the handle if available
                          if self.h5_file_handle:
                               h5_file = self.h5_file_handle
                          else:
                               # This is problematic - non-lazy should have a handle or pre-loaded data
                               raise RuntimeError(f"Group '{group_name}' not pre-loaded and no HDF5 handle available in non-lazy mode.")
                     else:
                          preloaded_group_data = self.data[group_name]
                          for name in self.data_keys.keys():
                              # Handle potential missing keys during pre-load
                              if preloaded_group_data.get(name) is not None:
                                   # Data is already numpy array from pre-load
                                   sample[name] = torch.from_numpy(preloaded_group_data[name])
                              else:
                                   sample[name] = None # Or handle error

                          # Apply transforms and convert types after loading all parts from pre-loaded data
                          for name, tensor in sample.items():
                              if tensor is not None:
                                  # Infer type (simple heuristic, may need refinement)
                                  # Add other potential integer keys from your config/preprocessing
                                  if name in ['phoneme', 'duration', 'phone_sequence', 'initial_duration_sequence']:
                                      sample[name] = tensor.long()
                                  else: # Assuming mel, f0 are float types
                                      sample[name] = tensor.float()
                          if self.transform:
                              sample = self.transform(sample) # Transform the dict
                          return sample # Return early if data was successfully retrieved from pre-loaded cache

                # If we reach here in non-lazy mode, it means pre-loaded data was missing for the group,
                # but we might have fallen back to using self.h5_file_handle
                if self.h5_file_handle:
                    h5_file = self.h5_file_handle
                elif self.h5_file_handle: # Use the open handle if data wasn't pre-loaded
                    h5_file = self.h5_file_handle
                else:
                    # This case shouldn't happen if init logic is correct
                    raise RuntimeError("Dataset not properly initialized for non-lazy loading.")

            # --- Common logic for lazy loading or using open handle ---
            if h5_file is None:
                 raise RuntimeError("HDF5 file handle not available.")

            # --- Access data within the group ---
            group_name = self.sample_keys[idx]
            group = h5_file[group_name]

            for name, h5_key in self.data_keys.items():
                if h5_key in group:
                    data_np = group[h5_key][()] # Read data from the group dataset


                    # Convert to tensor with appropriate type
                    # Add other potential integer keys from your config/preprocessing
                    if name in ['phoneme', 'duration', 'phone_sequence', 'initial_duration_sequence']:
                        sample[name] = torch.from_numpy(data_np).long()
                    else: # Assuming float types (mel, f0)
                        sample[name] = torch.from_numpy(data_np).float()
                else:
                    print(f"Warning: Key '{h5_key}' not found in group '{group_name}' for index {idx}.")
                    sample[name] = None # Handle missing key

            if self.transform:
                sample = self.transform(sample) # Apply transform to the dictionary

            return sample

        except Exception as e:
            print(f"Error loading index {idx} from {self.h5_path}: {e}")
            # Return None or raise error depending on desired behavior
            # Returning None requires handling in collate_fn
            return None

    def close(self):
        """Close the HDF5 file handle if it was opened by this instance (non-lazy)."""
        if self.h5_file_handle:
            self.h5_file_handle.close()
            self.h5_file_handle = None
        # Note: Lazy loaded files are closed by H5FileManager in teardown

# Removed MelSpectrogramDataset and VariableLengthMelDataset as H5Dataset now handles multiple keys
# and variable length logic will be primarily in collate_fn.
# Transposition/channel adding logic might need to be added back into H5Dataset.__getitem__
# or handled within the model/transforms if required by specific model architectures.

def create_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths."""
    if max_len is None:
        max_len = lengths.max().item()
    # Create range tensor (0, 1, ..., max_len-1)
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0) # Shape (1, MaxLen)
    # Compare lengths (Batch, 1) with range (1, MaxLen) -> broadcasts to (Batch, MaxLen)
    mask = ids < lengths.unsqueeze(1)
    return mask

def collate_fn_pad(batch):
    """
    Custom collate function for TransformerSVS.
    Pads sequences ('mel', 'f0', 'energy', 'phoneme', 'duration') and creates attention masks.
    - 'mel', 'f0', 'energy': Frame-level (Batch, Features, Time) or (Batch, Time) -> Padded to max Time
    - 'phoneme', 'duration': Phoneme-level (Batch, SeqLen) -> Padded to max SeqLen
    """
    # Filter out None items (samples that failed to load)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if batch is empty after filtering

    batch_size = len(batch)
    keys = batch[0].keys()
    padded_batch = {key: [] for key in keys}
    phone_lengths = []
    mel_lengths = []

    # Define keys needing padding and their type
    frame_keys = ['mel', 'f0', 'energy'] # Padded based on time dimension
    phone_keys = ['phoneme', 'duration'] # Padded based on sequence length

    # Determine max lengths
    max_mel_len = 0
    max_phone_len = 0
    for item in batch:
        if 'mel' in item: # Use mel length as the reference for frame-level sequences
            # Mel shape is (Time, Features)
            max_mel_len = max(max_mel_len, item['mel'].shape[0]) # Use shape[0] for time
        if 'phoneme' in item: # Use phoneme length as the reference for phone-level sequences
            max_phone_len = max(max_phone_len, item['phoneme'].shape[0])

    # Pad each item in the batch
    for item in batch:
        # Store original lengths (FIXED)
        if 'mel' in item: mel_lengths.append(item['mel'].shape[0]) # Use shape[0] for time
        if 'phoneme' in item: phone_lengths.append(item['phoneme'].shape[0])

        for key in keys:
            data = item[key]
            pad_value = 0.0 # Default padding value
            if key in frame_keys:
                target_len = max_mel_len
                # Data shape is (Time, Features) or (Time, 1)
                current_len = data.shape[0] # Use shape[0] for time
                if current_len < target_len:
                    padding_size = target_len - current_len
                    # Manual padding for clarity
                    if data.dim() > 1: # e.g., mel (Time, Features)
                        padded_data = torch.full((target_len,) + data.shape[1:], pad_value, dtype=data.dtype, device=data.device)
                        padded_data[:current_len, ...] = data
                    else: # e.g., f0, energy (Time,)
                        padded_data = torch.full((target_len,), pad_value, dtype=data.dtype, device=data.device)
                        padded_data[:current_len] = data
                elif current_len > target_len: # Should not happen if max_mel_len is correct
                    # Slice time dim
                    padded_data = data[:target_len, ...] if data.dim() > 1 else data[:target_len]
                else:
                    padded_data = data
            elif key in phone_keys:
                target_len = max_phone_len
                current_len = data.shape[0]
                # Use padding index 0 for phoneme IDs
                pad_value = 0 if key == 'phoneme' else 0.0
                if current_len < target_len:
                    padding_size = target_len - current_len
                    padded_data = torch.nn.functional.pad(data, (0, padding_size), mode='constant', value=pad_value)
                elif current_len > target_len: # Should not happen if max_phone_len is correct
                    padded_data = data[:target_len]
                else:
                    padded_data = data
            else: # Data doesn't need padding based on keys (e.g., scalar values)
                padded_data = data

            padded_batch[key].append(padded_data)

    # Stack tensors for each key
    final_batch = {}
    for key in keys:
        try:
            final_batch[key] = torch.stack(padded_batch[key])
        except Exception as e:
            print(f"Error stacking key '{key}': {e}")
            # Print shapes for debugging
            for i, t in enumerate(padded_batch[key]):
                print(f"  Item {i} shape: {t.shape}")
            raise e

    # Create masks and add lengths
    phone_lengths_tensor = torch.tensor(phone_lengths, dtype=torch.long)
    mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long)

    final_batch['phone_lengths'] = phone_lengths_tensor
    final_batch['mel_lengths'] = mel_lengths_tensor
    final_batch['src_mask'] = create_mask(phone_lengths_tensor, max_phone_len) # Mask for encoder input
    final_batch['mel_mask'] = create_mask(mel_lengths_tensor, max_mel_len)   # Mask for decoder output / variance adaptor input

    return final_batch

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)
        
        self.h5_path = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
        # Get all data keys from config
        self.data_keys = {
            'mel': config['data']['mel_key'],
            # 'phoneme': config['data']['phoneme_key'], # Original - Loads frame-level phonemes
            'phoneme': 'phone_sequence', # Corrected - Loads phone-level sequence IDs
            'duration': config['data']['duration_key'],
            'f0': config['data']['f0_key'],
            'energy': config['data']['energy_key'] # Added energy key
            # Optionally load frame-level phonemes under a different key if needed elsewhere
            # 'frame_phoneme': config['data']['phoneme_key'],
        }
        self.lazy_load = config['data'].get('lazy_load', True)
        self.max_samples = config['data'].get('max_samples', None)
        self.sample_percentage = config['data'].get('sample_percentage', None)
        
        # Handle variable length inputs
        # Variable length is now implicitly handled by the collate_fn
        # self.variable_length = config['data'].get('variable_length', False)

        # max_frames might still be useful for limiting sequence length during loading if desired
        # Calculate max time frames for 10 seconds of audio (example)
        # if 'max_audio_length' in config['audio']:
        #     self.max_audio_length = config['audio']['max_audio_length']
        #     sample_rate = config['audio']['sample_rate']
        #     hop_length = config['audio']['hop_length']
        #     self.max_frames = math.ceil(self.max_audio_length * sample_rate / hop_length)
        # else:
        #     self.max_frames = config['model'].get('time_frames', None) # Default to None if not specified
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.vocab_size = None # Initialize vocab_size
        
    def setup(self, stage=None):
        if self.dataset is None:
            # Instantiate the refactored H5Dataset
            self.dataset = H5Dataset(
                h5_path=self.h5_path,
                data_keys=self.data_keys,
                lazy_load=self.lazy_load
                # Add transforms here if needed
            )

            # --- Read vocab_size from HDF5 attribute ---
            h5_file = None
            try:
                # Use the file manager to get the handle
                h5_manager = H5FileManager.get_instance()
                h5_file = h5_manager.get_file(self.h5_path) # File is kept open by manager
                if 'vocab_size' in h5_file.attrs:
                    self.vocab_size = int(h5_file.attrs['vocab_size'])
                    print(f"Read vocab_size from HDF5 attribute: {self.vocab_size}")
                else:
                    raise ValueError(f"'vocab_size' attribute not found in HDF5 file: {self.h5_path}. "
                                     "Ensure it was saved during preprocessing.")
            except Exception as e:
                 # Ensure file handle is managed correctly even on error if needed,
                 # though H5FileManager should handle closing eventually.
                 print(f"Error reading vocab_size from HDF5 file: {e}")
                 raise e # Re-raise the exception to stop execution
            # Note: File remains open via H5FileManager if lazy loading, closed in teardown.
            # If not lazy loading, H5Dataset handles its own handle.

            # --- Continue with dataset splitting ---
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
            persistent_workers=True if self.num_workers > 0 else False, # Avoid warning when num_workers=0
            collate_fn=collate_fn_pad # Use the new padding collate function
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False, # Avoid warning when num_workers=0
            collate_fn=collate_fn_pad # Use the new padding collate function
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()