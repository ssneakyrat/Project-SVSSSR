import warnings
# Filter the specific h5py UserWarning about HDF5 version mismatch
warnings.filterwarnings("ignore", message=r"h5py is running against HDF5.*when it was built against.*", category=UserWarning)

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
                                  # Add 'midi_pitch_estimated', 'voiced_mask', 'unvoiced_flag' to integer/bool types
                                  if name in ['phoneme', 'duration', 'phone_sequence', 'initial_duration_sequence', 'midi_pitch_estimated', 'voiced_mask', 'unvoiced_flag']:
                                      # Convert voiced_mask to bool, unvoiced_flag to float (for potential projection), others to long
                                      if name == 'voiced_mask':
                                          sample[name] = tensor.bool()
                                      elif name == 'unvoiced_flag':
                                           # Keep as float for potential linear projection later
                                           sample[name] = tensor.float()
                                      else:
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
                    # Special handling for scalar original_unpadded_length
                    if name == 'original_unpadded_length':
                         # Read the scalar value directly
                         scalar_value = group[h5_key][()]
                         # Store as a Python int or a 0-dim tensor. Let's use int for now.
                         # inference.py already converts it to a tensor later.
                         sample[name] = int(scalar_value)
                    else:
                         # Existing logic for array types
                         data_np = group[h5_key][()] # Read data from the group dataset
                         # Convert to tensor with appropriate type
                         # Add other potential integer keys from your config/preprocessing
                         # Add 'midi_pitch_estimated', 'voiced_mask', 'unvoiced_flag' to integer/bool types
                         if name in ['phoneme', 'duration', 'phone_sequence', 'initial_duration_sequence', 'midi_pitch_estimated', 'voiced_mask', 'unvoiced_flag']:
                             # Convert voiced_mask to bool, unvoiced_flag to float, others to long
                             if name == 'voiced_mask':
                                 sample[name] = torch.from_numpy(data_np).bool()
                             elif name == 'unvoiced_flag':
                                  # Keep as float for potential linear projection later
                                  sample[name] = torch.from_numpy(data_np).float()
                             else:
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

def collate_fn_pad(batch):
    """
    Custom collate function to pad sequences in a batch of dictionaries.
    Handles padding for keys like 'mel', 'f0', 'midi_pitch_estimated', 'phoneme', 'duration'.
    Expands phoneme IDs ('phoneme') using 'duration' into frame-level 'phone_label'.
    Assumes 'mel', 'f0', 'midi_pitch_estimated', 'phone_label' are frame-level (variable length in time dim).
    Assumes 'phoneme', 'duration' are phoneme-level (variable length in sequence dim).
    """
    # Filter out None items (samples that failed to load)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if batch is empty after filtering

    batch_size = len(batch)
    # Get keys from the first valid sample
    keys = list(batch[0].keys())
    # Add the key for the expanded frame-level phoneme labels
    keys_in_batch = keys + ['phone_label']
    padded_batch = {key: [] for key in keys_in_batch}
    lengths = {key: [] for key in keys} # Store original lengths of loaded data

    # Determine max lengths for each sequence type
    max_len = {}
    # Determine max lengths for frame-level and phoneme-level sequences
    frame_level_keys = ['mel', 'f0', 'midi_pitch_estimated', 'voiced_mask', 'unvoiced_flag'] # Add voiced_mask, unvoiced_flag
    # 'phoneme' key holds frame-level IDs loaded from HDF5, remove it from here
    phoneme_level_keys = ['duration', 'phone_sequence_ids'] # Keys that are phoneme-level
    max_len_frame = 0
    max_len_phoneme = 0

    # Calculate max frame length based on frame-level keys
    for key in frame_level_keys:
        if key in keys: # Check if the key exists in the loaded data
            current_max_for_key = 0
            for i, item in enumerate(batch):
                # Check if the key exists for the item and if it's not None
                data_element = item.get(key)
                if data_element is None:
                    print(f"DEBUG: collate_fn_pad - Item index {i} has None for key '{key}'. Item keys: {list(item.keys())}")
                    # Decide how to handle: skip? use 0? For now, just report.
                    # We'll rely on current_max_for_key not being updated for this item.
                elif not hasattr(data_element, 'shape'):
                    print(f"DEBUG: collate_fn_pad - Item index {i}, key '{key}' is not None but has no 'shape'. Type: {type(data_element)}")
                    # Handle as above.
                else:
                    try:
                        # Use shape[0] for time dimension length
                        current_max_for_key = max(current_max_for_key, data_element.shape[0])
                    except IndexError:
                         print(f"DEBUG: collate_fn_pad - Item index {i}, key '{key}' has shape {data_element.shape}, but failed to access shape[-1].")

            max_len_frame = max(max_len_frame, current_max_for_key)

    # Calculate max phoneme sequence length
    for key in phoneme_level_keys:
        if key in keys:
             max_len_phoneme = max(max_len_phoneme, max(item[key].shape[0] for item in batch))

    # The target length for frame-level expansion will be max_len_frame
    max_len['phone_label'] = max_len_frame # Expanded phonemes match frame length
    for key in frame_level_keys:
        if key in keys:
            max_len[key] = max_len_frame
    for key in phoneme_level_keys:
        if key in keys:
            max_len[key] = max_len_phoneme

    # Pad each item in the batch
    # Process each item in the batch: expand phonemes, pad sequences
    for item in batch:
        # --- Expand Phoneme IDs to Frame Level ---
        # Load frame-level IDs (might not be needed here, but keeping for context)
        # frame_level_phoneme_ids = item['phoneme']
        # Load the PHONEME-LEVEL IDs and DURATIONS
        phoneme_level_ids = item['phone_sequence_ids']
        durations = item['duration']
        target_frame_len = max_len_frame # Target length is max frame length in batch
        # Ensure durations are positive and sum matches target_frame_len (or handle mismatch)
        # Note: Preprocessing should ideally ensure sum(durations) == target_frames
        # Add a check/warning here if needed.
        durations = torch.clamp(durations, min=0) # Ensure no negative durations
        total_duration = durations.sum()

        expanded_phonemes_list = []
        if total_duration > 0:
             # Use repeat_interleave for efficient expansion
             try:
                  # Use the phoneme-level IDs here
                  expanded_phonemes = torch.repeat_interleave(phoneme_level_ids, durations)
             except RuntimeError as e:
                  print(f"RuntimeError during repeat_interleave: {e}")
                  print(f"Phoneme IDs: {phoneme_ids.shape}, Durations: {durations.shape}, Sum: {total_duration}")
                  # Fallback or error handling: create zeros
                  expanded_phonemes = torch.zeros(target_frame_len, dtype=torch.long)

             current_expanded_len = expanded_phonemes.shape[0]

             # Pad or truncate the expanded sequence to match target_frame_len
             if current_expanded_len < target_frame_len:
                  padding_size = target_frame_len - current_expanded_len
                  expanded_phonemes = torch.nn.functional.pad(expanded_phonemes, (0, padding_size), mode='constant', value=0) # Pad with 0 (PAD symbol)
             elif current_expanded_len > target_frame_len:
                  expanded_phonemes = expanded_phonemes[:target_frame_len]

        else: # Handle case where total duration is zero
             expanded_phonemes = torch.zeros(target_frame_len, dtype=torch.long)
        padded_batch['phone_label'].append(expanded_phonemes)
        # Note: We don't store length for 'phone_label' as it's always max_len_frame

        # --- Pad Original Loaded Data ---
        for key in keys: # Iterate through original keys loaded from HDF5
            data = item[key]
            # Use shape[0] for time dimension length for frame-level keys
            current_len = data.shape[0] if key in frame_level_keys else data.shape[0]
            target_len = max_len.get(key, current_len) # Use max_len calculated earlier
            # Determine pad value (0 for IDs/indices, 0.0 for floats, False for bool)
            if key == 'voiced_mask':
                pad_value = False # Pad bool with False
            elif key == 'unvoiced_flag':
                pad_value = 0.0 # Pad float flag with 0.0
            elif key in ['phoneme', 'duration', 'midi_pitch_estimated']:
                pad_value = 0 # Pad integer IDs with 0
            else: # mel, f0
                pad_value = 0.0 # Pad float features with 0.0

            if current_len < target_len:
                padding_size = target_len - current_len
                if key in frame_level_keys: # Pad time dimension (first)
                    # Shape is likely (T, F) or (T,)
                    pad_dims = (0, padding_size) # Pad the first dimension (time)
                    if data.dim() > 1: # Handle feature dimensions (e.g., mel [T, F], f0 [T, 1])
                         # Need to pad the first dimension, not the last
                         # Example: (T, F) -> pad T dim -> (0, 0, 0, padding_size) incorrect
                         # Correct padding for (T, F) should be ((0, padding_size), (0, 0))
                         # torch.nn.functional.pad expects (pad_left, pad_right, pad_top, pad_bottom, ...)
                         # For (T, F), we want (0, 0, 0, padding_size) -> pads last dim (F) then second-to-last (T)
                         # Let's rethink padding for multi-dim frame-level
                         pad_shape = list(data.shape)
                         pad_shape[0] = padding_size # Pad the time dimension
                         padding_tensor = torch.full(pad_shape, pad_value, dtype=data.dtype)
                         padded_data = torch.cat((data, padding_tensor), dim=0)
                    else: # Handle 1D case (T,)
                         padded_data = torch.nn.functional.pad(data, (0, padding_size), mode='constant', value=pad_value)
                elif key in phoneme_level_keys: # Pad sequence dimension
                    padded_data = torch.nn.functional.pad(data, (0, padding_size), mode='constant', value=pad_value)
                else: # Data doesn't need padding based on keys
                    padded_data = data
            elif current_len > target_len: # Truncate if needed
                 if key in frame_level_keys:
                      padded_data = data[:target_len, ...] # Truncate first dim (time)
                 elif key in phoneme_level_keys:
                      padded_data = data[:target_len] # Truncate first dim (sequence)
                 else:
                      padded_data = data
            else: # No padding needed
                padded_data = data

            padded_batch[key].append(padded_data)
            lengths[key].append(current_len) # Store original length

    # Stack tensors for each key
    final_batch = {}
    # Stack tensors for the final batch, renaming keys as needed for the model
    final_batch = {}
    rename_map = {
        'mel': 'mel_spec', # Rename 'mel' to 'mel_spec'
        'f0': 'f0',
        'midi_pitch_estimated': 'midi_label', # Rename 'midi_pitch_estimated' to 'midi_label'
        'phone_label': 'phone_label', # Keep expanded phoneme label key
        'duration': 'phone_duration', # Rename original duration key
        'voiced_mask': 'voiced_mask', # Keep voiced_mask key
        'unvoiced_flag': 'unvoiced_flag' # Keep unvoiced_flag key
        # 'phoneme' key (original phoneme sequence) is not explicitly renamed/included
        # unless needed later.
    }

    for key in keys_in_batch: # Iterate through all keys including 'phone_label'
        try:
            stacked_tensor = torch.stack(padded_batch[key])
            # Rename key if needed for model input compatibility
            final_key = rename_map.get(key, key) # Use renamed key or original if not in map
            final_batch[final_key] = stacked_tensor
        except Exception as e:
            print(f"Error stacking key '{key}': {e}")
            for i, t in enumerate(padded_batch[key]):
                print(f"  Item {i} shape: {t.shape}")
            raise e

    # Add frame lengths (originally mel lengths) for masking in the model
    if 'mel' in lengths: # Use original 'mel' key for length lookup
         final_batch['length'] = torch.tensor(lengths['mel'], dtype=torch.long) # Use 'length' as key like in model

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
        # Get all data keys from config, ensuring they match HDF5 and model needs
        self.data_keys = {
            'mel': config['data']['mel_key'],           # e.g., 'mel_spectrogram'
            'f0': config['data']['f0_key'],             # e.g., 'f0_contour'
            'phoneme': config['data']['phoneme_key'],   # e.g., 'phone_sequence' (FRAME-level IDs)
            'duration': config['data']['duration_key'], # e.g., 'adjusted_durations' (PHONEME-level durations)
            'phone_sequence_ids': 'phone_sequence_ids', # Key for PHONEME-level IDs
            'midi_pitch_estimated': 'midi_pitch_estimated', # Key for estimated MIDI pitch
            'voiced_mask': 'voiced_mask', # Key for the voiced mask
            'unvoiced_flag': 'unvoiced_flag' # Key for the unvoiced flag
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

            # --- Perform dataset subsetting and splitting only if not already done ---
            if self.train_dataset is None or self.val_dataset is None:
                print("Performing train/validation split...") # Add log message
                full_dataset_size = len(self.dataset)
                dataset_to_split = self.dataset # Start with the full dataset

                # Apply subsetting if configured
                subset_size = full_dataset_size
                if self.max_samples and self.max_samples > 0:
                    subset_size = min(self.max_samples, full_dataset_size)
                elif self.sample_percentage and 0.0 < self.sample_percentage <= 1.0:
                    subset_size = int(full_dataset_size * self.sample_percentage)

                if subset_size < full_dataset_size:
                    print(f"Applying subset: Using {subset_size} samples out of {full_dataset_size}.")
                    generator_subset = torch.Generator().manual_seed(42) # Use separate generator for subsetting consistency
                    indices = torch.randperm(full_dataset_size, generator=generator_subset)[:subset_size].tolist()
                    dataset_to_split = torch.utils.data.Subset(self.dataset, indices)
                # else: # Use the full dataset if no subsetting applied
                    # dataset_to_split = self.dataset # Already assigned

                # Perform the split on the (potentially subsetted) dataset
                dataset_size = len(dataset_to_split)
                val_size = int(dataset_size * self.validation_split)
                train_size = dataset_size - val_size

                # Ensure sizes are valid
                if train_size <= 0 or val_size <= 0:
                     raise ValueError(f"Calculated train_size ({train_size}) or val_size ({val_size}) is not positive. Check dataset size ({dataset_size}) and validation_split ({self.validation_split}).")


                generator_split = torch.Generator().manual_seed(42) # Use separate generator for splitting consistency

                self.train_dataset, self.val_dataset = random_split(
                    dataset_to_split,
                    [train_size, val_size],
                    generator=generator_split
                )
                print(f"Split complete: Train size={train_size}, Validation size={val_size}")
            else:
                 print("Train/validation datasets already exist, skipping split.") # Add log message
    
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