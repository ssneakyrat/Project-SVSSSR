import warnings
# Filter the specific h5py UserWarning about HDF5 version mismatch
warnings.filterwarnings("ignore", message=r"h5py is running against HDF5.*when it was built against.*", category=UserWarning)

import logging # Import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np
# Removed torchaudio import as it's no longer needed here for loading
import os # For path joining
import math

logger = logging.getLogger(__name__) # Module-level logger

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
    Dataset to load multiple data types (e.g., mel, phonemes, f0, durations, audio)
    from different keys within a single HDF5 file.
    Assumes all datasets referenced by keys have the same number of samples.
    """
    # Removed raw_audio_dir and target_sample_rate from __init__
    def __init__(self, h5_path, data_keys, transform=None, lazy_load=True):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            data_keys (dict): Dictionary mapping descriptive names to HDF5 dataset keys
                              (e.g., {'mel': 'mel_spectrogram', 'audio_waveform': 'raw_audio'}).
            transform (callable, optional): Optional transform to be applied on a sample.
            lazy_load (bool): Whether to load data on demand or all at once.
        """
        self.h5_path = h5_path
        # Ensure 'audio_waveform' maps to the correct HDF5 key ('raw_audio')
        self.data_keys = data_keys
        if 'audio_waveform' not in self.data_keys:
             logger.warning("Key 'audio_waveform' not found in data_keys. Audio will not be loaded.")
             # Or map it explicitly if not passed:
             # self.data_keys['audio_waveform'] = 'raw_audio' # Ensure this key exists in HDF5

        self.transform = transform
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
                self.h5_file_handle = h5py.File(h5_path, 'r')
                h5_file = self.h5_file_handle

            self.sample_keys = list(h5_file.keys())
            if not self.sample_keys:
                raise ValueError(f"No sample groups found in HDF5 file: {h5_path}")
            self.length = len(self.sample_keys)

            # Sanity check keys in the first sample group
            if self.sample_keys:
                first_group_name = self.sample_keys[0]
                try:
                    first_group = h5_file[first_group_name]
                    # Check all expected keys, including the one for raw audio
                    all_expected_h5_keys = list(self.data_keys.values())
                    for h5_key in all_expected_h5_keys:
                        if h5_key not in first_group:
                            logger.warning(f"HDF5 key '{h5_key}' not found in the first sample group '{first_group_name}'. Assuming it exists in others.")
                except KeyError:
                     logger.warning(f"Could not access first group '{first_group_name}' for key check in {h5_path}.")

            # Pre-load data if not lazy loading
            if not self.lazy_load:
                logger.info(f"Pre-loading data from {self.h5_path}...")
                self.data = {}
                for group_name in tqdm(self.sample_keys, desc="Pre-loading HDF5"):
                    self.data[group_name] = {}
                    group = h5_file[group_name]
                    for name, h5_key in self.data_keys.items():
                        if h5_key in group:
                            self.data[group_name][name] = group[h5_key][()]
                        else:
                            self.data[group_name][name] = None
                            logger.warning(f"Key '{h5_key}' not found in group '{group_name}' during pre-loading.")
                logger.info("Pre-loading complete.")

        except Exception as e:
            if self.h5_file_handle:
                self.h5_file_handle.close()
            raise e

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Loads a single sample (group) from the HDF5 file."""
        sample = {}
        h5_file = None

        try:
            group_name = self.sample_keys[idx] # Get group name first

            # Get HDF5 file handle
            if self.lazy_load:
                h5_manager = H5FileManager.get_instance()
                h5_file = h5_manager.get_file(self.h5_path)
            else:
                # Use pre-loaded data if available, otherwise use persistent handle
                if hasattr(self, 'data') and group_name in self.data:
                    preloaded_group_data = self.data[group_name]
                    # Load from pre-loaded cache
                    for name, h5_key in self.data_keys.items():
                        cached_data = preloaded_group_data.get(name)
                        if cached_data is not None:
                            # Convert numpy array from cache to tensor
                            tensor = torch.from_numpy(cached_data)
                            # Apply type conversion based on key name
                            if name in ['phoneme', 'duration', 'phone_sequence', 'initial_duration_sequence', 'midi_pitch_estimated', 'phone_sequence_ids']:
                                sample[name] = tensor.long()
                            elif name == 'voiced_mask':
                                sample[name] = tensor.bool()
                            elif name == 'unvoiced_flag':
                                sample[name] = tensor.float()
                            elif name == 'audio_waveform': # Handle audio type
                                sample[name] = tensor.float()
                            else: # Assume float for others (mel, f0)
                                sample[name] = tensor.float()
                        else:
                            sample[name] = None # Mark as None if missing in cache

                    # Apply transform and return early
                    if self.transform:
                        sample = self.transform(sample)
                    return sample
                elif self.h5_file_handle:
                    # Pre-load failed or wasn't done, use the handle
                    h5_file = self.h5_file_handle
                else:
                    raise RuntimeError("Dataset not properly initialized for non-lazy loading without pre-loaded data.")

            # --- Common logic for lazy loading or using the persistent handle ---
            if h5_file is None:
                 raise RuntimeError("HDF5 file handle not available.")

            group = h5_file[group_name]

            # Load data directly from HDF5 group
            for name, h5_key in self.data_keys.items():
                if h5_key in group:
                    # Special handling for scalar original_unpadded_length
                    if name == 'original_unpadded_length':
                         scalar_value = group[h5_key][()]
                         sample[name] = int(scalar_value)
                    else:
                         data_np = group[h5_key][()]
                         tensor = torch.from_numpy(data_np)
                         # Apply type conversion based on key name
                         if name in ['phoneme', 'duration', 'phone_sequence', 'initial_duration_sequence', 'midi_pitch_estimated', 'phone_sequence_ids']:
                             sample[name] = tensor.long()
                         elif name == 'voiced_mask':
                             sample[name] = tensor.bool()
                         elif name == 'unvoiced_flag':
                             sample[name] = tensor.float()
                         elif name == 'audio_waveform': # Handle audio type
                             sample[name] = tensor.float()
                         else: # Assume float for others (mel, f0)
                             sample[name] = tensor.float()
                else:
                    logger.warning(f"Key '{h5_key}' not found in group '{group_name}' for index {idx}.")
                    sample[name] = None # Mark as None if key is missing

            # --- Removed filesystem audio loading block ---

            # Apply any specified transforms
            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            logger.error(f"Error loading index {idx} from {self.h5_path}: {e}", exc_info=True)
            return None # Return None requires handling in collate_fn

    def close(self):
        """Close the HDF5 file handle if it was opened by this instance (non-lazy)."""
        if self.h5_file_handle:
            self.h5_file_handle.close()
            self.h5_file_handle = None
        # Note: Lazy loaded files are closed by H5FileManager in teardown


def collate_fn_pad(batch):
    """
    Collate function for DataLoader. Handles:
    1. Filtering None items (errors during __getitem__).
    2. Determining max sequence lengths for frame-level, phoneme-level, and audio data.
    3. Expanding phoneme-level IDs and durations to frame-level phoneme labels.
    4. Padding all sequences in the batch to the maximum length for their type.
    5. Stacking tensors for each key.
    6. Renaming keys to match model input expectations.
    """
    # Filter out None items (samples that failed to load)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if batch is empty after filtering

    batch_size = len(batch)
    # Get keys from the first valid sample (assuming all valid samples have the same keys)
    keys = list(batch[0].keys())
    # Define potential keys that might be added during collation
    added_keys = ['phone_label']
    # Combine original keys and potentially added keys
    keys_in_batch = list(set(keys + added_keys)) # Use set to avoid duplicates

    padded_batch = {key: [] for key in keys_in_batch}
    # Initialize lengths dict only for keys present in the original loaded data
    lengths = {key: [] for key in keys}

    # Determine max lengths for each sequence type
    max_len = {} # Store max lengths for padding
    # Categorize keys based on their expected level (frame, phoneme, audio)
    frame_level_keys = ['mel', 'f0', 'midi_pitch_estimated', 'voiced_mask', 'unvoiced_flag'] # Features aligned with mel frames
    audio_keys = ['audio_waveform'] # Raw audio has its own length
    phoneme_level_keys = ['duration', 'phone_sequence_ids'] # Keys that are phoneme-level

    max_len_frame = 0
    max_len_audio = 0
    max_len_phoneme = 0

    # --- Determine Max Lengths ---
    # Calculate max frame length
    for key in frame_level_keys:
        if key in keys:
            current_max_for_key = 0
            for item in batch:
                data_element = item.get(key)
                if data_element is not None and hasattr(data_element, 'shape') and data_element.dim() > 0:
                    try: current_max_for_key = max(current_max_for_key, data_element.shape[0])
                    except IndexError: logger.debug(f"collate_fn_pad - Item key '%s' shape %s error.", key, data_element.shape)
            max_len_frame = max(max_len_frame, current_max_for_key)

    # Calculate max audio length
    for key in audio_keys:
        if key in keys:
             current_max_for_key = 0
             for item in batch:
                 data_element = item.get(key)
                 if data_element is not None and hasattr(data_element, 'shape') and data_element.dim() > 0:
                     try: current_max_for_key = max(current_max_for_key, data_element.shape[0])
                     except IndexError: logger.debug(f"collate_fn_pad - Audio key '%s' shape %s error.", key, data_element.shape)
             max_len_audio = max(max_len_audio, current_max_for_key)

    # Calculate max phoneme sequence length
    for key in phoneme_level_keys:
        if key in keys:
             current_max_for_key = 0
             for item in batch:
                 data_element = item.get(key)
                 if data_element is not None and hasattr(data_element, 'shape') and data_element.dim() > 0:
                     try: current_max_for_key = max(current_max_for_key, data_element.shape[0])
                     except IndexError: logger.debug(f"collate_fn_pad - Phoneme key '%s' shape %s error.", key, data_element.shape)
             max_len_phoneme = max(max_len_phoneme, current_max_for_key)


    # Set target lengths for padding
    max_len['phone_label'] = max_len_frame
    for key in frame_level_keys:
        if key in keys: max_len[key] = max_len_frame
    for key in phoneme_level_keys:
        if key in keys: max_len[key] = max_len_phoneme
    for key in audio_keys:
        if key in keys: max_len[key] = max_len_audio

    # --- Process Each Sample ---
    for item in batch:
        # --- Expand Phoneme IDs to Frame Level ---
        if 'phone_sequence_ids' in item and item['phone_sequence_ids'] is not None and \
           'duration' in item and item['duration'] is not None:
            phoneme_level_ids = item['phone_sequence_ids']
            durations = item['duration']
            target_frame_len = max_len_frame

            durations = torch.clamp(durations, min=0)
            total_duration = durations.sum()
            expanded_phonemes = torch.zeros(target_frame_len, dtype=torch.long)

            if total_duration > 0 and len(phoneme_level_ids) == len(durations):
                 try:
                      expanded_phonemes_unpadded = torch.repeat_interleave(phoneme_level_ids, durations)
                      current_expanded_len = expanded_phonemes_unpadded.shape[0]
                      if current_expanded_len < target_frame_len:
                           padding_size = target_frame_len - current_expanded_len
                           expanded_phonemes = torch.nn.functional.pad(expanded_phonemes_unpadded, (0, padding_size), mode='constant', value=0)
                      elif current_expanded_len > target_frame_len:
                           expanded_phonemes = expanded_phonemes_unpadded[:target_frame_len]
                      else:
                           expanded_phonemes = expanded_phonemes_unpadded
                 except RuntimeError as e:
                      logger.error(f"RuntimeError during repeat_interleave: {e}", exc_info=True)
                      logger.error(f"Phoneme IDs shape: {phoneme_level_ids.shape}, Durations shape: {durations.shape}, Sum: {total_duration}")
            elif total_duration > 0:
                 logger.warning(f"Mismatch between phoneme ID count ({len(phoneme_level_ids)}) and duration count ({len(durations)}). Using zero padding for phone_label.")

            padded_batch['phone_label'].append(expanded_phonemes)
        else:
            logger.warning("Missing 'phone_sequence_ids' or 'duration' in item. Padding 'phone_label' with zeros.")
            padded_batch['phone_label'].append(torch.zeros(max_len_frame, dtype=torch.long))


        # --- Pad Original Loaded Data ---
        for key in keys: # Iterate through original keys loaded from HDF5/audio
            data = item.get(key)

            if data is None:
                padded_batch[key].append(None)
                lengths[key].append(0)
                continue

            # --- ADD THIS CHECK ---
            if not isinstance(data, torch.Tensor):
                 # Skip padding logic for non-tensor data
                 padded_batch[key].append(None) # Append None for consistency
                 lengths[key].append(0)         # Length is 0 for non-sequence data
                 continue
            # --- END ADDED CHECK ---

            current_len = data.shape[0] if data.dim() > 0 else 0
            target_len = max_len.get(key, current_len)

            if key == 'voiced_mask': pad_value = False
            elif key == 'unvoiced_flag': pad_value = 0.0
            elif key in ['phoneme', 'duration', 'midi_pitch_estimated', 'phone_sequence_ids']: pad_value = 0
            elif key in audio_keys: pad_value = 0.0
            else: pad_value = 0.0 # mel, f0

            if current_len < target_len:
                padding_size = target_len - current_len
                if data.dim() > 1:
                     pad_shape = list(data.shape); pad_shape[0] = padding_size
                     padding_tensor = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)
                     padded_data = torch.cat((data, padding_tensor), dim=0)
                elif data.dim() == 1:
                     padded_data = torch.nn.functional.pad(data, (0, padding_size), mode='constant', value=pad_value)
                else: padded_data = data
            elif current_len > target_len:
                 padded_data = data[:target_len, ...] if data.dim() > 1 else data[:target_len]
            else: padded_data = data

            padded_batch[key].append(padded_data)
            lengths[key].append(current_len)

    # --- Stack and Rename ---
    final_batch = {}
    rename_map = {
        'mel': 'mel_spec',
        'f0': 'f0',
        'midi_pitch_estimated': 'midi_label',
        'phone_label': 'phone_label',
        'duration': 'phone_duration',
        'voiced_mask': 'voiced_mask',
        'unvoiced_flag': 'unvoiced_flag',
        'audio_waveform': 'audio_waveform'
    }

    for key in keys_in_batch:
        items_to_stack = [item for item in padded_batch[key] if item is not None]
        if not items_to_stack: continue

        try:
            first_shape = items_to_stack[0].shape
            if not all(t.shape == first_shape for t in items_to_stack):
                 logger.error(f"Cannot stack tensors for key '{key}' due to shape mismatch after padding.")
                 for i, t in enumerate(items_to_stack): logger.error(f"  Item {i} shape: {t.shape}")
                 continue
        except IndexError:
             logger.error(f"Error checking shapes for key '{key}'. Items might be 0-dim or invalid.")
             continue

        try:
            stacked_tensor = torch.stack(items_to_stack)
            final_key = rename_map.get(key, key)
            final_batch[final_key] = stacked_tensor
        except Exception as e:
            logger.error(f"Error stacking key '{key}': {e}", exc_info=True)
            for i, t in enumerate(items_to_stack): logger.error(f"  Item {i} shape: {t.shape}")
            raise e

    # Add frame lengths
    if 'mel' in lengths and 'mel_spec' in final_batch:
         valid_mel_lengths = [l for i, l in enumerate(lengths['mel']) if padded_batch['mel'][i] is not None]
         if len(valid_mel_lengths) == final_batch['mel_spec'].shape[0]:
              final_batch['length'] = torch.tensor(valid_mel_lengths, dtype=torch.long)
         else: logger.warning("Mismatch mel lengths vs stacked tensor size. Not adding 'length'.")

    # Add audio lengths
    if 'audio_waveform' in lengths and 'audio_waveform' in final_batch:
         valid_audio_lengths = [l for i, l in enumerate(lengths['audio_waveform']) if padded_batch['audio_waveform'][i] is not None]
         if len(valid_audio_lengths) == final_batch['audio_waveform'].shape[0]:
              final_batch['audio_length'] = torch.tensor(valid_audio_lengths, dtype=torch.long)
         else: logger.warning("Mismatch audio lengths vs stacked tensor size. Not adding 'audio_length'.")

    # Add original unpadded lengths if available
    if 'original_unpadded_length' in keys:
        original_lengths_list = [item.get('original_unpadded_length') for item in batch if item is not None and isinstance(item.get('original_unpadded_length'), int)]
        if len(original_lengths_list) == batch_size:
             final_batch['original_length'] = torch.tensor(original_lengths_list, dtype=torch.long)
        else:
             logger.warning("Mismatch or missing 'original_unpadded_length' in batch items. Not adding 'original_length'.")


    return final_batch


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)

        self.h5_path = os.path.join(config['data']['bin_dir'], config['data']['bin_file'])
        # Removed raw_audio_dir and target_sample_rate attributes

        # Define keys expected from HDF5, including the raw audio
        self.data_keys = {
            'mel': config['data']['mel_key'],
            'f0': config['data']['f0_key'],
            'duration': config['data']['duration_key'],
            'phone_sequence_ids': 'phone_sequence_ids', # Key for PHONEME-level IDs
            'midi_pitch_estimated': 'midi_pitch_estimated',
            'voiced_mask': 'voiced_mask',
            'unvoiced_flag': 'unvoiced_flag',
            'audio_waveform': 'raw_audio', # Map internal key to HDF5 key 'raw_audio'
            'original_unpadded_length': 'original_unpadded_length' # Include original length key
            # 'phoneme' (frame-level) is generated in collate_fn
        }
        self.lazy_load = config['data'].get('lazy_load', True)
        self.max_samples = config['data'].get('max_samples', None)
        self.sample_percentage = config['data'].get('sample_percentage', None)

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.vocab_size = None

    def setup(self, stage=None):
        # Called on every GPU

        # Ensure HDF5 file exists
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 file not found at: {self.h5_path}")
        # Removed raw_audio_dir check

        if not self.dataset:
            # Pass only necessary args to H5Dataset
            self.dataset = H5Dataset(
                h5_path=self.h5_path,
                data_keys=self.data_keys,
                lazy_load=self.lazy_load
                # Removed raw_audio_dir and target_sample_rate
            )

            # --- Handle max_samples and sample_percentage ---
            num_total_samples = len(self.dataset)
            num_samples_to_use = num_total_samples

            if self.sample_percentage is not None and 0 < self.sample_percentage <= 1:
                num_samples_to_use = int(num_total_samples * self.sample_percentage)
                logger.info(f"Using {self.sample_percentage*100:.2f}% of the dataset: {num_samples_to_use} samples.")
            elif self.max_samples is not None and self.max_samples > 0:
                num_samples_to_use = min(self.max_samples, num_total_samples)
                logger.info(f"Using a maximum of {self.max_samples} samples (found {num_samples_to_use}).")

            if num_samples_to_use < num_total_samples:
                indices = torch.randperm(num_total_samples)[:num_samples_to_use].tolist()
                full_dataset = self.dataset
                self.dataset = torch.utils.data.Subset(full_dataset, indices)
                logger.info(f"Created subset with {len(self.dataset)} samples.")
            # --- End Handle max_samples/sample_percentage ---


            # Determine vocab size dynamically
            first_sample_idx = 0
            if isinstance(self.dataset, torch.utils.data.Subset):
                 if len(self.dataset.indices) > 0:
                      first_sample_idx = self.dataset.indices[0]
                      # Access original dataset to get item for vocab check
                      original_dataset = self.dataset.dataset
                      first_sample = original_dataset[first_sample_idx]
                 else: first_sample = None
            else:
                 if len(self.dataset) > 0: first_sample = self.dataset[0]
                 else: first_sample = None

            if first_sample and 'phone_sequence_ids' in first_sample and first_sample['phone_sequence_ids'] is not None:
                 max_phoneme_id = torch.max(first_sample['phone_sequence_ids']).item()
                 self.vocab_size = max_phoneme_id + 1
                 logger.info(f"Dynamically determined vocab_size: {self.vocab_size} (based on max ID {max_phoneme_id} in first sample)")
                 if self.config['model'].get('vocab_size') is None:
                     self.config['model']['vocab_size'] = self.vocab_size
                     logger.info("Updated config['model']['vocab_size']")
            else:
                 if self.config['model'].get('vocab_size') is None:
                      logger.error("Could not determine vocab_size dynamically and it's not set in config.")
                      self.vocab_size = 100 # Fallback
                      self.config['model']['vocab_size'] = self.vocab_size
                      logger.warning(f"Using fallback vocab_size: {self.vocab_size}")
                 else:
                      self.vocab_size = self.config['model']['vocab_size']
                      logger.info(f"Using pre-configured vocab_size: {self.vocab_size}")


            # Split dataset
            num_samples = len(self.dataset)
            num_val = int(num_samples * self.validation_split)
            num_train = num_samples - num_val
            self.train_dataset, self.val_dataset = random_split(self.dataset, [num_train, num_val])
            logger.info(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")

    def train_dataloader(self):
        if not self.train_dataset: self.setup()
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn_pad,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        if not self.val_dataset: self.setup()
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn_pad,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def teardown(self, stage=None):
        # Close HDF5 files managed by H5FileManager
        h5_manager = H5FileManager.get_instance()
        h5_manager.close_all()
        logger.info("Closed HDF5 file handles managed by H5FileManager.")
        # Also close the handle if non-lazy loading was used directly
        if hasattr(self, 'dataset') and self.dataset:
             original_dataset = self.dataset.dataset if isinstance(self.dataset, torch.utils.data.Subset) else self.dataset
             if isinstance(original_dataset, H5Dataset):
                  original_dataset.close()