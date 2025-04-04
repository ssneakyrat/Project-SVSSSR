# preprocess.py

import os
import glob
import yaml
import numpy as np
import librosa
import pyworld as pw
import h5py
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import math
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
def load_config(config_path='config/model.yaml'):
    """Loads configuration from a YAML file."""
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# --- Label Parsing and Duration Adjustment ---
def parse_lab_file(lab_path):
    """Parses a .lab file and returns timestamps, phones, and unique phones."""
    lab_entries = []
    unique_phones = set()
    max_end_time = 0.0
    try:
        with open(lab_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start_time_str, end_time_str, phone = parts
                    try:
                        start_sec = float(start_time_str)
                        end_sec = float(end_time_str)
                        if start_sec < 0 or end_sec < start_sec:
                             logging.warning(f"Skipping invalid time entry in {lab_path}: {line.strip()}")
                             continue
                        lab_entries.append((start_sec, end_sec, phone))
                        unique_phones.add(phone)
                        max_end_time = max(max_end_time, end_sec)
                    except ValueError:
                         logging.warning(f"Skipping non-numeric time entry in {lab_path}: {line.strip()}")
                         continue
                else:
                     logging.warning(f"Skipping malformed line in {lab_path}: {line.strip()}")

    except Exception as e:
        logging.error(f"Error parsing lab file {lab_path}: {e}")
        return [], [], set()
    return lab_entries, unique_phones, max_end_time

def adjust_durations(phones, durations, target_frames, silence_symbols={'sil', 'sp', '<SIL>'}):
    """Adjusts phone durations to match target_frames, handling gaps/mismatches."""
    total_lab_frames = sum(durations)
    discrepancy = target_frames - total_lab_frames
    adjusted_durations = list(durations) # Make a copy

    if discrepancy == 0:
        return adjusted_durations

    logging.debug(f"Adjusting durations: Target={target_frames}, Original={total_lab_frames}, Discrepancy={discrepancy}")

    eligible_indices = [i for i, p in enumerate(phones) if p not in silence_symbols and durations[i] > 0]
    total_eligible_duration = sum(durations[i] for i in eligible_indices)

    if not eligible_indices or total_eligible_duration <= 0:
        # If only silence or zero-duration phones, distribute among all positive duration phones
        eligible_indices = [i for i, d in enumerate(durations) if d > 0]
        total_eligible_duration = sum(durations[i] for i in eligible_indices)
        if not eligible_indices:
             logging.warning("Cannot adjust durations: No positive duration phones found.")
             # Return original durations, might lead to length mismatch later
             return adjusted_durations


    if discrepancy > 0: # Need to add frames
        distributed_frames = 0
        for i in eligible_indices:
            # Avoid division by zero if total_eligible_duration is somehow zero
            if total_eligible_duration > 0:
                add = round(discrepancy * (durations[i] / total_eligible_duration))
            else:
                add = round(discrepancy / len(eligible_indices)) # Distribute equally if no duration info
            adjusted_durations[i] += add
            distributed_frames += add

        # Handle rounding remainder
        remainder = discrepancy - distributed_frames
        if remainder != 0 and eligible_indices: # Ensure eligible_indices is not empty
             # Add/subtract remainder to/from the longest eligible phone
             longest_eligible_idx = max(eligible_indices, key=lambda i: adjusted_durations[i])
             adjusted_durations[longest_eligible_idx] += remainder
             logging.debug(f"Applied remainder {remainder} to phone index {longest_eligible_idx}")


    else: # Need to remove frames (discrepancy is negative)
        removed_frames = 0
        # Sort by duration descending to prioritize shrinking longer phones
        sorted_eligible_indices = sorted(eligible_indices, key=lambda i: durations[i], reverse=True)

        frames_to_remove = abs(discrepancy)
        current_removed = 0

        # First pass: proportional removal
        temp_removed = {}
        if total_eligible_duration > 0: # Avoid division by zero
            for i in sorted_eligible_indices:
                 reduction = math.floor(frames_to_remove * (durations[i] / total_eligible_duration))
                 # Ensure duration doesn't go below 1
                 reduction = min(reduction, adjusted_durations[i] - 1)
                 if reduction > 0:
                     temp_removed[i] = reduction
                     current_removed += reduction
        else: # If total duration is 0, cannot remove proportionally
             logging.warning("Total eligible duration is 0, cannot perform proportional removal.")


        # Apply removals from first pass
        for i, reduction in temp_removed.items():
            adjusted_durations[i] -= reduction

        # Second pass: handle remainder by removing 1 frame at a time from longest eligible
        remainder_to_remove = frames_to_remove - current_removed
        idx_pool = list(sorted_eligible_indices) # Use the sorted list

        while remainder_to_remove > 0 and idx_pool:
            # Find longest eligible phone *that can still be reduced*
            target_idx = -1
            max_dur = 0
            best_i_in_pool = -1 # Index within idx_pool
            for pool_idx, original_idx in enumerate(idx_pool):
                 if adjusted_durations[original_idx] > 1:
                     if adjusted_durations[original_idx] > max_dur:
                         max_dur = adjusted_durations[original_idx]
                         target_idx = original_idx
                         best_i_in_pool = pool_idx

            if target_idx != -1:
                adjusted_durations[target_idx] -= 1
                remainder_to_remove -= 1
                # Remove from pool if it becomes 1, so we don't try reducing it further
                if adjusted_durations[target_idx] == 1:
                    idx_pool.pop(best_i_in_pool)
            else:
                # No more phones can be reduced
                logging.warning(f"Could not remove all required frames. {remainder_to_remove} frames remaining.")
                break # Exit loop

    # Final check
    final_sum = sum(adjusted_durations)
    if final_sum != target_frames:
        logging.warning(f"Duration adjustment resulted in {final_sum} frames, expected {target_frames}. Attempting final clamp.")
        # As a fallback, clamp to target_frames (might truncate/extend last phone)
        diff = target_frames - final_sum
        if diff > 0: # Need to add frames
            adjusted_durations[-1] += diff
        elif diff < 0: # Need to remove frames
            if adjusted_durations[-1] > abs(diff):
                 adjusted_durations[-1] += diff # diff is negative here
            else:
                 logging.error(f"Cannot clamp duration by reducing last phone (duration {adjusted_durations[-1]}, need to remove {abs(diff)}). Final sum will be incorrect.")
                 # Consider distributing the reduction across other phones if this happens often

    return adjusted_durations

# --- Feature Extraction ---
def extract_features(audio_path, config):
    """Extracts log Mel spectrogram, log F0, and RMS energy."""
    sr = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    win_length = config['audio']['win_length']
    n_fft = config['audio']['n_fft']
    n_mels = config['audio']['n_mels']
    fmin = config['audio']['fmin']
    fmax = config['audio']['fmax']
    f0_min = config['audio']['f0_min']
    f0_max = config['audio']['f0_max']

    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr)
        # Use float32 for librosa features, convert to float64 for pyworld if needed later
        audio_float32 = audio.astype(np.float32)

        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_float32, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        # Log-scale Mel spectrogram (common practice)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        log_mel_spectrogram = log_mel_spectrogram.T # Shape (T, n_mels)
        mel_frames = log_mel_spectrogram.shape[0]

        # Extract F0 using PyWorld
        audio_float64 = audio.astype(np.float64) # PyWorld requires float64
        _f0, t = pw.dio(audio_float64, sr, f0_floor=f0_min, f0_ceil=f0_max,
                        frame_period=hop_length * 1000 / sr)
        f0 = pw.stonemask(audio_float64, _f0, t, sr) # Refine F0

        # Ensure F0 length matches Mel length
        if len(f0) < mel_frames:
            f0 = np.pad(f0, (0, mel_frames - len(f0)), mode='constant', constant_values=0)
        elif len(f0) > mel_frames:
            f0 = f0[:mel_frames]

        # Log-scale F0, handle zeros
        voiced_mask = f0 > 1e-8 # Use a small epsilon to avoid log(0)
        log_f0 = np.zeros_like(f0)
        log_f0[voiced_mask] = np.log(f0[voiced_mask])
        log_f0 = log_f0[:, np.newaxis] # Shape (T, 1)

        # Extract RMS Energy
        energy = librosa.feature.rms(y=audio_float32, frame_length=win_length, hop_length=hop_length)[0]

        # Ensure energy length matches Mel length
        if len(energy) < mel_frames:
            energy = np.pad(energy, (0, mel_frames - len(energy)), mode='constant', constant_values=0)
        elif len(energy) > mel_frames:
            energy = energy[:mel_frames]

        energy = energy[:, np.newaxis] # Shape (T, 1)

        return log_mel_spectrogram, log_f0, energy # Return log_mel, log_f0, energy

    except Exception as e:
        logging.error(f"Error extracting features for {audio_path}: {e}")
        # Return None for all expected outputs on error
        return None, None, None

# --- Padding/Truncating ---
def pad_or_truncate(data, target_length, pad_value=0):
    """Pads or truncates data along the first axis."""
    current_length = data.shape[0]
    if current_length == target_length:
        return data
    elif current_length > target_length:
        return data[:target_length]
    else: # current_length < target_length
        padding_shape = list(data.shape)
        padding_shape[0] = target_length - current_length
        padding = np.full(padding_shape, pad_value, dtype=data.dtype)
        return np.concatenate((data, padding), axis=0)

# --- Visualization ---
def visualize_alignment(mel, f0, phone_ids, durations, id_to_phone, save_path):
    """Generates and saves an alignment plot using UNNORMALIZED data."""
    logging.info(f"Generating alignment plot: {save_path}")
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # 1. Mel Spectrogram (Unnormalized)
        img = axes[0].imshow(mel.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_ylabel('Mel Bin')
        axes[0].set_title('Mel Spectrogram (Unnormalized)')

        # 2. F0 Contour (Unnormalized, convert log-F0 back for plotting)
        # Handle unvoiced frames (where log_f0 might be 0 or negative infinity)
        voiced_f0 = np.exp(f0) # Convert back from log
        voiced_f0[f0 <= np.log(1e-8)] = np.nan # Set unvoiced frames to NaN for plotting gaps
        axes[1].plot(voiced_f0, label='F0', marker='.', markersize=2, linestyle='')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_title('F0 Contour (Unnormalized)')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # 3. Phone Alignment
        axes[2].set_title('Phone Alignment')
        axes[2].set_ylabel('Phone ID')
        axes[2].set_xlabel('Frame')
        axes[2].grid(True, linestyle='--', alpha=0.6)

        current_frame = 0
        yticks = []
        yticklabels = []
        phone_boundaries = [0]
        for phone_id, duration in zip(phone_ids, durations):
            phone_symbol = id_to_phone.get(str(phone_id), '?') # Ensure lookup uses string if keys are strings
            center_frame = current_frame + duration / 2
            axes[2].text(center_frame, phone_id, phone_symbol, ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            current_frame += duration
            phone_boundaries.append(current_frame)
            if phone_id not in yticks:
                 yticks.append(phone_id)
                 yticklabels.append(phone_symbol)


        # Draw vertical lines for phone boundaries
        for boundary in phone_boundaries:
             axes[2].axvline(x=boundary, color='r', linestyle='--', linewidth=0.8)

        # Set y-ticks for phones
        if yticks:
            sorted_unique_yticks = sorted(list(set(yticks)))
            axes[2].set_yticks(sorted_unique_yticks)
            axes[2].set_yticklabels([id_to_phone.get(str(y), '?') for y in sorted_unique_yticks]) # Ensure lookup uses string
            axes[2].set_ylim(min(sorted_unique_yticks)-1 if sorted_unique_yticks else -1, max(sorted_unique_yticks)+1 if sorted_unique_yticks else 1)


        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Alignment plot saved to {save_path}")

    except Exception as e:
        logging.error(f"Error generating visualization: {e}")


# --- Main Processing Function ---
def preprocess_data(config_path='config/model.yaml'):
    """Main function to run the preprocessing pipeline with normalization."""
    config = load_config(config_path)
    data_config = config['data']
    audio_config = config['audio']

    # Get config values
    raw_dir = data_config['raw_dir']
    bin_dir = data_config['bin_dir']
    bin_file = data_config['bin_file']
    min_phones_in_lab = data_config.get('min_phones_in_lab', 5)
    output_hdf5_path = os.path.join(bin_dir, bin_file)
    stats_file_path = os.path.join(bin_dir, "norm_stats.json") # File to save stats

    lab_dir = os.path.join(raw_dir, 'lab')
    wav_dir = os.path.join(raw_dir, 'wav')

    # Create output directory if it doesn't exist
    os.makedirs(bin_dir, exist_ok=True)

    # Find file pairs
    lab_files = glob.glob(os.path.join(lab_dir, '*.lab'))
    file_pairs = []
    for lab_path in lab_files:
        base_filename = os.path.splitext(os.path.basename(lab_path))[0]
        wav_path = os.path.join(wav_dir, base_filename + '.wav')
        if os.path.exists(wav_path):
            file_pairs.append({'id': base_filename, 'lab': lab_path, 'wav': wav_path})
        else:
            logging.warning(f"Corresponding WAV file not found for {lab_path}, skipping.")

    if not file_pairs:
        logging.error("No valid (lab, wav) file pairs found. Exiting.")
        return

    logging.info(f"Found {len(file_pairs)} file pairs to process.")

    # --- Pass 1: Extract all features and calculate normalization stats ---
    logging.info("Starting Pass 1: Extracting features and calculating normalization stats...")
    all_log_mels = []
    all_log_f0s_voiced = []
    all_energies = [] # Added list for energy

    for pair in tqdm(file_pairs, desc="Pass 1: Extracting Features"):
        log_mel, log_f0, energy = extract_features(pair['wav'], config) # Unpack energy
        # Check all returned features
        if log_mel is not None and log_f0 is not None and energy is not None:
            # Store features for stat calculation
            all_log_mels.append(log_mel)
            all_energies.append(energy) # Store energy
            # Only consider voiced frames for F0 stats (using log_f0 shape for mask)
            voiced_mask = log_f0 > np.log(1e-8) # Recreate mask based on log_f0
            if np.any(voiced_mask):
                 all_log_f0s_voiced.append(log_f0[voiced_mask])
        else:
            logging.warning(f"Feature extraction failed for {pair['id']}, excluding from stats.")
            # Mark pair as failed? Or handle later. For now, just exclude from stats.
            pair['failed_extraction'] = True


    if not all_log_mels or not all_log_f0s_voiced:
         logging.error("No valid features extracted in Pass 1. Cannot calculate stats. Exiting.")
         return

    # Concatenate all features
    all_log_mels_np = np.concatenate(all_log_mels, axis=0)
    all_log_f0s_voiced_np = np.concatenate(all_log_f0s_voiced, axis=0)
    all_energies_np = np.concatenate(all_energies, axis=0) # Concatenate energy

    # Calculate Mel stats
    mel_mean = np.mean(all_log_mels_np, axis=0)
    mel_std = np.std(all_log_mels_np, axis=0)
    # Avoid division by zero for std dev
    mel_std[mel_std < 1e-5] = 1e-5

    # Calculate F0 stats (only on voiced frames)
    f0_mean = np.mean(all_log_f0s_voiced_np)
    f0_std = np.std(all_log_f0s_voiced_np)
    if f0_std < 1e-5:
        logging.warning(f"F0 standard deviation is very low ({f0_std}). Setting to 1e-5.")
        f0_std = 1e-5

    # Calculate Energy stats
    energy_mean = np.mean(all_energies_np)
    energy_std = np.std(all_energies_np)
    if energy_std < 1e-5:
        logging.warning(f"Energy standard deviation is very low ({energy_std}). Setting to 1e-5.")
        energy_std = 1e-5

    logging.info(f"Calculated Mel Mean shape: {mel_mean.shape}, Mel Std shape: {mel_std.shape}")
    logging.info(f"Calculated F0 Mean (voiced, log): {f0_mean:.4f}, F0 Std (voiced, log): {f0_std:.4f}")

    # Save stats
    norm_stats = {
        'mel_mean': mel_mean.tolist(), # Convert to list for JSON
        'mel_std': mel_std.tolist(),
        'f0_mean': float(f0_mean),
        'f0_std': float(f0_std),
        'energy_mean': float(energy_mean), # Add energy stats
        'energy_std': float(energy_std)
    }
    try:
        with open(stats_file_path, 'w') as f_stats:
            json.dump(norm_stats, f_stats, indent=4)
        logging.info(f"Normalization stats saved to {stats_file_path}")
    except Exception as e:
        logging.error(f"Error saving normalization stats: {e}")
        # Continue processing, but warn user stats are not saved
        # Or decide to stop here? Let's stop for safety.
        raise

    # --- Pass 2: Process files, normalize, and prepare for saving ---
    logging.info("Starting Pass 2: Normalizing features and preparing data...")
    all_unique_phones = set(["<PAD>"]) # Add padding symbol initially
    processed_data = {} # Store results temporarily before HDF5 write
    first_file_data_for_vis = None # Store unnormalized data for the first valid file

    target_frames = math.ceil(audio_config['max_audio_length'] * audio_config['sample_rate'] / audio_config['hop_length'])
    logging.info(f"Target frames per file: {target_frames}")

    for pair in tqdm(file_pairs, desc="Pass 2: Processing & Normalizing"):
        base_filename = pair['id']
        lab_path = pair['lab']
        wav_path = pair['wav']

        # Skip if feature extraction failed in Pass 1
        if pair.get('failed_extraction', False):
            continue

        logging.debug(f"Processing: {base_filename}")

        # 1. Re-extract Mel, F0, and Energy
        log_mel, log_f0, energy = extract_features(wav_path, config) # Unpack energy
        if log_mel is None or log_f0 is None or energy is None: # Check all features
            logging.warning(f"Skipping {base_filename} due to feature extraction error in Pass 2.")
            continue

        original_mel_frames = log_mel.shape[0] # Frames before padding/truncation

        # 2. Normalize features
        norm_log_mel = (log_mel - mel_mean) / mel_std
        # Recreate voiced mask based on log_f0 for normalization
        voiced_mask = log_f0 > np.log(1e-8)
        norm_log_f0 = np.zeros_like(log_f0) # Initialize with zeros (unvoiced)
        if np.any(voiced_mask): # Apply normalization only to voiced frames
            norm_log_f0[voiced_mask] = (log_f0[voiced_mask] - f0_mean) / f0_std
        # Normalize energy
        norm_energy = (energy - energy_mean) / energy_std

        # 3. Pad/Truncate NORMALIZED features to target_frames
        # Pad normalized mel with 0? Or with normalized mean? Padding with 0 is common.
        padded_norm_mel = pad_or_truncate(norm_log_mel, target_frames, pad_value=0.0)
        padded_norm_f0 = pad_or_truncate(norm_log_f0, target_frames, pad_value=0.0)
        padded_norm_energy = pad_or_truncate(norm_energy, target_frames, pad_value=0.0) # Pad energy

        # 4. Parse Labels (get timestamps and phones)
        lab_entries, unique_phones_in_file, max_end_time = parse_lab_file(lab_path)
        if not lab_entries:
             logging.warning(f"Skipping {base_filename} due to empty or invalid lab file.")
             continue

        # Check minimum number of phones
        if len(lab_entries) < min_phones_in_lab:
             logging.warning(f"Skipping {base_filename}: Found {len(lab_entries)} phones, required minimum {min_phones_in_lab}.")
             continue

        all_unique_phones.update(unique_phones_in_file)

        # 5. Calculate Initial Durations based on Scaled Timestamps
        initial_phones = []
        initial_durations = []
        if max_end_time <= 0:
             logging.warning(f"Max end time in lab file {lab_path} is 0 or negative. Cannot scale durations. Skipping {base_filename}.")
             continue
        if original_mel_frames <= 0:
             logging.warning(f"Original mel frames for {wav_path} is 0. Cannot scale durations. Skipping {base_filename}.")
             continue

        scale_factor = original_mel_frames / max_end_time
        logging.debug(f"{base_filename}: Original Frames={original_mel_frames}, Max Lab Time={max_end_time:.4f}, Scale Factor={scale_factor:.4f}")

        last_calculated_end_frame = 0
        for start_sec, end_sec, phone in lab_entries:
            start_frame = round(start_sec * scale_factor)
            end_frame = round(end_sec * scale_factor)
            start_frame = max(start_frame, last_calculated_end_frame)
            end_frame = max(end_frame, start_frame)

            duration = end_frame - start_frame
            if duration > 0:
                initial_phones.append(phone)
                initial_durations.append(duration)
                last_calculated_end_frame = end_frame
            else:
                 logging.warning(f"Skipping zero duration phone '{phone}' after scaling in {base_filename} ({start_sec:.3f}-{end_sec:.3f}) -> Frames ({start_frame}-{end_frame})")

        if not initial_phones:
             logging.warning(f"Skipping {base_filename} as no phones with positive duration remained after scaling.")
             continue

        # 6. Adjust Scaled Durations to Match Target Frames
        adjusted_durations = adjust_durations(initial_phones, initial_durations, target_frames)

        # Check if adjustment was successful
        if sum(adjusted_durations) != target_frames:
             logging.error(f"Final duration adjustment failed for {base_filename}. Sum={sum(adjusted_durations)}, Target={target_frames}. Skipping file.")
             continue

        # 7. Store results for this file
        processed_data[base_filename] = {
            'mel': padded_norm_mel, # Store normalized mel
            'f0': padded_norm_f0,   # Store normalized f0
            'energy': padded_norm_energy, # Store normalized energy
            'phone_sequence': initial_phones,
            'duration_sequence': np.array(adjusted_durations, dtype=np.int32),
            'initial_duration_sequence': np.array(initial_durations, dtype=np.int32) # Keep for potential analysis/vis
        }

        # Store unnormalized data for the *first* successfully processed file for visualization
        if first_file_data_for_vis is None:
             first_file_data_for_vis = {
                 'mel': pad_or_truncate(log_mel, target_frames, pad_value=0.0), # Pad original log_mel
                 'f0': pad_or_truncate(log_f0, target_frames, pad_value=0.0),   # Pad original log_f0
                 'phone_sequence': initial_phones,
                 'initial_duration_sequence': np.array(initial_durations, dtype=np.int32)
             }


    # --- Post-processing ---
    if not processed_data:
        logging.error("No files were processed successfully in Pass 2. HDF5 file will not be created.")
        return

    # 8. Create Phone Map
    sorted_phones = sorted(list(all_unique_phones))
    phone_to_id = {phone: i for i, phone in enumerate(sorted_phones)}
    # Ensure id_to_phone uses string keys if phone_to_id does, for JSON compatibility and lookup consistency
    id_to_phone = {str(i): phone for phone, i in phone_to_id.items()}
    vocab_size = len(phone_to_id)
    logging.info(f"Created phone map with {vocab_size} symbols: {phone_to_id}")


    # 9. Save to HDF5
    logging.info(f"Saving processed data to HDF5 file: {output_hdf5_path}")
    try:
        with h5py.File(output_hdf5_path, 'w') as f:
            # Save phone map and vocab size as attributes
            f.attrs['phone_map'] = json.dumps(phone_to_id)
            f.attrs['id_to_phone_map'] = json.dumps(id_to_phone)
            f.attrs['vocab_size'] = vocab_size
            # Save normalization stats as attributes
            f.attrs['mel_mean'] = mel_mean
            f.attrs['mel_std'] = mel_std
            f.attrs['f0_mean'] = f0_mean
            f.attrs['f0_std'] = f0_std
            f.attrs['energy_mean'] = energy_mean # Add energy stats
            f.attrs['energy_std'] = energy_std
            logging.info(f"Saved vocab_size ({vocab_size}) and normalization stats as HDF5 attributes.")

            # Save data for each file
            for base_filename, data in tqdm(processed_data.items(), desc="Saving to HDF5"):
                group = f.create_group(base_filename)

                # Convert phone sequence to IDs
                phone_sequence_ids = np.array([phone_to_id[p] for p in data['phone_sequence']], dtype=np.int32)

                # Create frame-level phone IDs from adjusted durations
                frame_level_phone_ids = np.repeat(phone_sequence_ids, data['duration_sequence'])
                # Ensure final length is exactly target_frames
                if len(frame_level_phone_ids) != target_frames:
                     logging.warning(f"Frame-level phone ID length mismatch for {base_filename}. Got {len(frame_level_phone_ids)}, expected {target_frames}. Clamping.")
                     frame_level_phone_ids = pad_or_truncate(frame_level_phone_ids, target_frames, pad_value=phone_to_id["<PAD>"])


                # Save datasets (using keys from config)
                # TODO: Add 'energy_key' to config/model.yaml
                energy_key = data_config.get('energy_key', 'energy') # Default key if not in config yet
                group.create_dataset(data_config['mel_key'], data=data['mel'], compression="gzip") # Normalized mel
                group.create_dataset(data_config['f0_key'], data=data['f0'], compression="gzip")   # Normalized f0
                group.create_dataset(energy_key, data=data['energy'], compression="gzip") # Normalized energy
                group.create_dataset(data_config['phoneme_key'], data=frame_level_phone_ids, compression="gzip") # Frame-level IDs
                group.create_dataset(data_config['duration_key'], data=data['duration_sequence'], compression="gzip") # Adjusted durations per phone
                group.create_dataset('phone_sequence', data=phone_sequence_ids, compression="gzip") # IDs corresponding to durations
                # group.create_dataset('initial_duration_sequence', data=data['initial_duration_sequence'], compression="gzip") # Optional: Save initial durations
        logging.info("HDF5 file saved successfully.")

    except Exception as e:
        logging.error(f"Error saving HDF5 file: {e}")
        raise

    # 10. Visualization (using the first file's UNNORMALIZED data)
    if first_file_data_for_vis:
        first_filename = list(processed_data.keys())[0] # Get the ID of the first successfully processed file
        logging.info(f"Generating visualization for example file: {first_filename}")
        vis_save_path = os.path.join(bin_dir, f"{first_filename}_alignment_check.png")

        # Get phone IDs corresponding to the duration sequence for visualization
        example_phone_ids = np.array([phone_to_id[p] for p in first_file_data_for_vis['phone_sequence']], dtype=np.int32)

        visualize_alignment(
            first_file_data_for_vis['mel'], # Unnormalized log-mel
            first_file_data_for_vis['f0'],  # Unnormalized log-f0
            example_phone_ids,
            first_file_data_for_vis['initial_duration_sequence'], # Use initial durations for alignment vis
            id_to_phone,
            vis_save_path
        )
    else:
        logging.warning("Skipping visualization as no data was processed successfully.")

    logging.info("Preprocessing finished.")


if __name__ == "__main__":
    # Example usage: Run preprocessing using the default config path
    try:
        preprocess_data()
    except Exception as e:
         logging.exception("Preprocessing failed with an error.") # Log full traceback