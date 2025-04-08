# preprocess.py

import os
import collections
import glob
import yaml
import numpy as np
import librosa
import pyworld as pw
import warnings
# Filter the specific h5py UserWarning about HDF5 version mismatch
warnings.filterwarnings("ignore", message=r"h5py is running against HDF5.*when it was built against.*", category=UserWarning)
import h5py
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import math
import logging
from utils.utils import setup_logging # Import setup_logging

logger = logging.getLogger(__name__) # Get module-level logger

# --- Configuration Loading ---
def load_config(config_path='config/model.yaml'):
    """Loads configuration from a YAML file."""
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
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
                             logger.warning(f"Skipping invalid time entry in {lab_path}: {line.strip()}")
                             continue
                        lab_entries.append((start_sec, end_sec, phone))
                        unique_phones.add(phone)
                        max_end_time = max(max_end_time, end_sec)
                    except ValueError:
                         logger.warning(f"Skipping non-numeric time entry in {lab_path}: {line.strip()}")
                         continue
                else:
                     logger.warning(f"Skipping malformed line in {lab_path}: {line.strip()}")

    except Exception as e:
        logger.error(f"Error parsing lab file {lab_path}: {e}")
        return [], [], set()
    return lab_entries, unique_phones, max_end_time

def adjust_durations(phones, durations, target_frames, silence_symbols={'sil', 'sp', '<SIL>'}):
    """Adjusts phone durations to match target_frames, handling gaps/mismatches."""
    total_lab_frames = sum(durations)
    discrepancy = target_frames - total_lab_frames
    adjusted_durations = list(durations) # Make a copy

    if discrepancy == 0:
        return adjusted_durations

    logger.debug(f"Adjusting durations: Target={target_frames}, Original={total_lab_frames}, Discrepancy={discrepancy}")

    eligible_indices = [i for i, p in enumerate(phones) if p not in silence_symbols and durations[i] > 0]
    total_eligible_duration = sum(durations[i] for i in eligible_indices)

    if not eligible_indices or total_eligible_duration <= 0:
        # If only silence or zero-duration phones, distribute among all positive duration phones
        eligible_indices = [i for i, d in enumerate(durations) if d > 0]
        total_eligible_duration = sum(durations[i] for i in eligible_indices)
        if not eligible_indices:
             logger.warning("Cannot adjust durations: No positive duration phones found.")
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
             logger.debug(f"Applied remainder {remainder} to phone index {longest_eligible_idx}")


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
             logger.warning("Total eligible duration is 0, cannot perform proportional removal.")


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
                logger.warning(f"Could not remove all required frames. {remainder_to_remove} frames remaining.")
                break # Exit loop

    # Final check
    final_sum = sum(adjusted_durations)
    if final_sum != target_frames:
        logger.warning(f"Duration adjustment resulted in {final_sum} frames, expected {target_frames}. Attempting final clamp.")
        # As a fallback, clamp to target_frames (might truncate/extend last phone)
        diff = target_frames - final_sum
        if diff > 0: # Need to add frames
            adjusted_durations[-1] += diff
        elif diff < 0: # Need to remove frames
            if adjusted_durations[-1] > abs(diff):
                 adjusted_durations[-1] += diff # diff is negative here
            else:
                 logger.error(f"Cannot clamp duration by reducing last phone (duration {adjusted_durations[-1]}, need to remove {abs(diff)}). Final sum will be incorrect.")
                 # Consider distributing the reduction across other phones if this happens often

    return adjusted_durations

# --- Feature Extraction ---
def extract_features(audio_path, config):
    """Extracts log Mel spectrogram, log F0, voiced mask, and estimated MIDI pitch."""
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
        audio = audio.astype(np.float64) # PyWorld requires float64

        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        # Log-scale Mel spectrogram (common practice)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        log_mel_spectrogram = log_mel_spectrogram.T # Shape (T, n_mels)

        # Extract F0 using PyWorld
        _f0, t = pw.dio(audio, sr, f0_floor=f0_min, f0_ceil=f0_max,
                        frame_period=hop_length * 1000 / sr)
        f0 = pw.stonemask(audio, _f0, t, sr) # Refine F0

        # Ensure F0 length matches Mel length
        mel_frames = log_mel_spectrogram.shape[0]
        if len(f0) < mel_frames:
            f0 = np.pad(f0, (0, mel_frames - len(f0)), mode='constant', constant_values=0)
        elif len(f0) > mel_frames:
            f0 = f0[:mel_frames]

        # Keep original F0 in Hz for MIDI conversion
        f0_hz = f0.copy()

        # Calculate voiced mask (use f0_hz)
        voiced_mask = f0_hz > 1e-8 # Use a small epsilon

        # Calculate log F0 for normalization/saving
        log_f0 = np.zeros_like(f0_hz)
        log_f0[voiced_mask] = np.log(f0_hz[voiced_mask])
        log_f0 = log_f0[:, np.newaxis] # Shape (T, 1)

        # Estimate MIDI pitch from F0 in Hz
        midi_pitch = np.zeros_like(f0_hz)
        voiced_f0_hz = f0_hz[voiced_mask]
        # MIDI formula: midi = 69 + 12 * log2(f/440)
        # Use np.log2 for base-2 logarithm
        # Add small epsilon to avoid log2(0) for safety, although voiced_mask should prevent this
        midi_pitch[voiced_mask] = 69 + 12 * np.log2(voiced_f0_hz / 440.0 + 1e-12)
        # Round to nearest integer MIDI note
        midi_pitch = np.round(midi_pitch).astype(int)
        # Clamp to valid MIDI range [0, 127], use 0 for unvoiced
        midi_pitch = np.clip(midi_pitch, 0, 127)
        midi_pitch[~voiced_mask] = 0 # Ensure unvoiced frames are 0
        midi_pitch = midi_pitch[:, np.newaxis] # Shape (T, 1)

        return log_mel_spectrogram, log_f0, voiced_mask.squeeze(), midi_pitch

    except Exception as e:
        logger.error(f"Error extracting features for {audio_path}: {e}")
        # Return None for all expected outputs on error
        return None, None, None, None # Added None for midi_pitch

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
def visualize_alignment(mel, f0, midi_pitch, phone_ids, durations, id_to_phone, save_path):
    """Generates and saves an alignment plot using UNNORMALIZED data, including estimated MIDI pitch."""
    logger.info(f"Generating alignment plot: {save_path}")
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True) # Increased subplots to 4, adjusted figsize

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

        # 3. Estimated MIDI Pitch
        # Plot MIDI pitch, setting 0 (unvoiced) to NaN for gaps
        midi_plot = midi_pitch.astype(float).squeeze() # Ensure float for NaN
        midi_plot[midi_plot == 0] = np.nan
        axes[2].plot(midi_plot, label='Est. MIDI Pitch', marker='.', markersize=2, linestyle='', color='green')
        axes[2].set_ylabel('MIDI Note #')
        axes[2].set_title('Estimated MIDI Pitch (0=Unvoiced)')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].set_ylim(bottom=0) # Start y-axis at 0

        # 4. Phone Alignment (Shifted index from 2 to 3)
        axes[3].set_title('Phone Alignment')
        axes[3].set_ylabel('Phone ID')
        axes[3].set_xlabel('Frame')
        axes[3].grid(True, linestyle='--', alpha=0.6)

        current_frame = 0
        yticks = []
        yticklabels = []
        phone_boundaries = [0]
        for phone_id, duration in zip(phone_ids, durations):
            phone_symbol = id_to_phone.get(str(phone_id), '?') # Ensure lookup uses string if keys are strings
            center_frame = current_frame + duration / 2
            axes[3].text(center_frame, phone_id, phone_symbol, ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            current_frame += duration
            phone_boundaries.append(current_frame)
            if phone_id not in yticks:
                 yticks.append(phone_id)
                 yticklabels.append(phone_symbol)


        # Draw vertical lines for phone boundaries
        for boundary in phone_boundaries:
             axes[3].axvline(x=boundary, color='r', linestyle='--', linewidth=0.8)

        # Set y-ticks for phones
        if yticks:
            sorted_unique_yticks = sorted(list(set(yticks)))
            axes[3].set_yticks(sorted_unique_yticks)
            axes[3].set_yticklabels([id_to_phone.get(str(y), '?') for y in sorted_unique_yticks]) # Ensure lookup uses string
            axes[3].set_ylim(min(sorted_unique_yticks)-1 if sorted_unique_yticks else -1, max(sorted_unique_yticks)+1 if sorted_unique_yticks else 1)


        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Alignment plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")


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
            logger.warning(f"Corresponding WAV file not found for %s, skipping.", lab_path)

    if not file_pairs:
        logger.error("No valid (lab, wav) file pairs found. Exiting.")
        return

    logger.info(f"Found %d file pairs to process.", len(file_pairs))

    # --- Pass 1: Extract all features and calculate normalization stats ---
    logger.info("Starting Pass 1: Extracting features and calculating normalization stats...")
    all_log_mels = []
    all_log_f0s_voiced = []

    for pair in tqdm(file_pairs, desc="Pass 1: Extracting Features"):
        # MIDI pitch not needed for stats, ignore it here
        log_mel, log_f0, voiced_mask, _ = extract_features(pair['wav'], config)
        if log_mel is not None and log_f0 is not None and voiced_mask is not None: # Check original features
            # Store features for stat calculation
            all_log_mels.append(log_mel)
            # Only consider voiced frames for F0 stats
            if np.any(voiced_mask):
                 all_log_f0s_voiced.append(log_f0[voiced_mask])
        else:
            logger.warning(f"Feature extraction failed for %s, excluding from stats.", pair['id'])
            # Mark pair as failed? Or handle later. For now, just exclude from stats.
            pair['failed_extraction'] = True


    if not all_log_mels or not all_log_f0s_voiced:
         logger.error("No valid features extracted in Pass 1. Cannot calculate stats. Exiting.")
         return

    # Concatenate all features
    all_log_mels_np = np.concatenate(all_log_mels, axis=0)
    all_log_f0s_voiced_np = np.concatenate(all_log_f0s_voiced, axis=0)

    # Calculate Mel stats
    mel_mean = np.mean(all_log_mels_np, axis=0)
    mel_std = np.std(all_log_mels_np, axis=0)
    # Avoid division by zero for std dev
    mel_std[mel_std < 1e-5] = 1e-5

    # Calculate F0 stats (only on voiced frames)
    f0_mean = np.mean(all_log_f0s_voiced_np)
    f0_std = np.std(all_log_f0s_voiced_np)
    if f0_std < 1e-5:
        logger.warning(f"F0 standard deviation is very low (%f). Setting to 1e-5.", f0_std)
        f0_std = 1e-5

    logger.info(f"Calculated Mel Mean shape: %s, Mel Std shape: %s", mel_mean.shape, mel_std.shape)
    logger.info(f"Calculated F0 Mean (voiced, log): %.4f, F0 Std (voiced, log): %.4f", f0_mean, f0_std)

    # Save stats
    norm_stats = {
        'mel_mean': mel_mean.tolist(), # Convert to list for JSON
        'mel_std': mel_std.tolist(),
        'f0_mean': f0_mean,
        'f0_std': f0_std
    }
    try:
        with open(stats_file_path, 'w') as f_stats:
            json.dump(norm_stats, f_stats, indent=4)
        logger.info(f"Normalization stats saved to %s", stats_file_path)
    except Exception as e:
        logger.error(f"Error saving normalization stats: %s", e)
        # Continue processing, but warn user stats are not saved
        # Or decide to stop here? Let's stop for safety.
        raise

    # --- Pass 2: Process files, normalize, and prepare for saving ---
    logger.info("Starting Pass 2: Normalizing features and preparing data...")
    all_unique_phones = set(["<PAD>"]) # Add padding symbol initially
    processed_data = {} # Store results temporarily before HDF5 write
    first_file_data_for_vis = None # Store unnormalized data for the first valid file

    target_frames = math.ceil(audio_config['max_audio_length'] * audio_config['sample_rate'] / audio_config['hop_length'])
    target_samples = target_frames * audio_config['hop_length'] # Calculate target samples for audio
    logger.info(f"Target frames per file: %d, Target samples: %d", target_frames, target_samples)

    for pair in tqdm(file_pairs, desc="Pass 2: Processing & Normalizing"):
        base_filename = pair['id']
        lab_path = pair['lab']
        wav_path = pair['wav']

        # Skip if feature extraction failed in Pass 1
        if pair.get('failed_extraction', False):
            continue

        logger.debug(f"Processing: %s", base_filename)

        # 1. Re-extract Mel and F0 (or retrieve if stored - re-extracting is simpler here)
        log_mel, log_f0, voiced_mask, midi_pitch = extract_features(wav_path, config)
        if log_mel is None or midi_pitch is None: # Check again just in case, include midi_pitch
            logger.warning(f"Skipping %s due to feature extraction error in Pass 2 (Mel or MIDI).", base_filename)
            continue

        original_mel_frames = log_mel.shape[0] # Frames before padding/truncation
        
        # 1.5 Load Raw Audio
        try:
            raw_audio, _ = librosa.load(wav_path, sr=audio_config['sample_rate'])
            raw_audio = raw_audio.astype(np.float32) # Ensure float32 for consistency
        except Exception as e:
            logger.error(f"Error loading raw audio for {wav_path}: {e}. Skipping file.")
            continue

        # 2. Normalize features
        norm_log_mel = (log_mel - mel_mean) / mel_std
        norm_log_f0 = np.zeros_like(log_f0) # Initialize with zeros (unvoiced)
        if np.any(voiced_mask): # Apply normalization only to voiced frames
            norm_log_f0[voiced_mask] = (log_f0[voiced_mask] - f0_mean) / f0_std

        # 3. Pad/Truncate NORMALIZED Mel, F0, and MIDI pitch to target_frames
        # Pad normalized mel with 0? Or with normalized mean? Padding with 0 is common.
        padded_norm_mel = pad_or_truncate(norm_log_mel, target_frames, pad_value=0.0)
        padded_norm_f0 = pad_or_truncate(norm_log_f0, target_frames, pad_value=0.0)
        padded_midi_pitch = pad_or_truncate(midi_pitch, target_frames, pad_value=0) # Pad MIDI with 0 (unvoiced)
        # Pad voiced mask (boolean -> int for padding, pad with 0/False)
        padded_voiced_mask = pad_or_truncate(voiced_mask.astype(np.uint8), target_frames, pad_value=0)
        # Create and pad unvoiced flag (1 for unvoiced, 0 for voiced)
        unvoiced_flag = (~voiced_mask).astype(np.uint8)
        padded_unvoiced_flag = pad_or_truncate(unvoiced_flag, target_frames, pad_value=0)
        # Pad/Truncate RAW AUDIO to target_samples
        padded_raw_audio = pad_or_truncate(raw_audio, target_samples, pad_value=0.0)

        # 4. Parse Labels (get timestamps and phones)
        lab_entries, unique_phones_in_file, max_end_time = parse_lab_file(lab_path)
        if not lab_entries:
             logger.warning(f"Skipping %s due to empty or invalid lab file.", base_filename)
             continue

        # Check minimum number of phones
        if len(lab_entries) < min_phones_in_lab:
             logger.warning(f"Skipping %s: Found %d phones, required minimum %d.", base_filename, len(lab_entries), min_phones_in_lab)
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

        # --- MODIFICATION START: Calculate, Round, and Borrow ---
        phones_sequence = []
        calculated_durations = []
        last_calculated_end_frame = 0

        # 1. Calculate scaled frames and durations for ALL phones
        for start_sec, end_sec, phone in lab_entries:
            start_frame = round(start_sec * scale_factor)
            end_frame = round(end_sec * scale_factor)
            # Ensure frames don't overlap due to rounding/tiny segments
            start_frame = max(start_frame, last_calculated_end_frame)
            end_frame = max(end_frame, start_frame)

            duration = end_frame - start_frame
            phones_sequence.append(phone)
            calculated_durations.append(duration)
            last_calculated_end_frame = end_frame

        # 2. Apply borrowing logic
        final_durations = list(calculated_durations) # Copy to modify
        num_phones = len(final_durations)
        for i in range(num_phones):
            if final_durations[i] <= 0: # Check for <= 0 just in case
                original_duration = final_durations[i]
                borrowed = False
                phone_symbol = phones_sequence[i]
                # Try borrowing from next
                if i + 1 < num_phones and final_durations[i+1] > 1:
                    final_durations[i] = 1
                    final_durations[i+1] -= 1
                    borrowed = True
                    logging.debug(f"Borrowing from next: Phone {i} ('{phone_symbol}') duration {original_duration} -> 1, Phone {i+1} ('{phones_sequence[i+1]}') duration {calculated_durations[i+1]} -> {final_durations[i+1]} in {base_filename}")
                # Else, try borrowing from previous
                elif i - 1 >= 0 and final_durations[i-1] > 1:
                    final_durations[i] = 1
                    final_durations[i-1] -= 1
                    borrowed = True
                    logging.debug(f"Borrowing from previous: Phone {i} ('{phone_symbol}') duration {original_duration} -> 1, Phone {i-1} ('{phones_sequence[i-1]}') duration {calculated_durations[i-1]} -> {final_durations[i-1]} in {base_filename}")

                # Else, force to 1
                if not borrowed:
                    final_durations[i] = 1
                    logging.warning(f"Forcing duration to 1 for phone {i} ('{phone_symbol}') (original: {original_duration}) in {base_filename} as neighbours cannot lend.")

        # 3. Populate initial_phones and initial_durations
        initial_phones = phones_sequence
        initial_durations = final_durations

        # Check if any durations are still invalid *after* borrowing (shouldn't happen)
        if any(d <= 0 for d in initial_durations):
             logging.error(f"Error: Found zero or negative duration after borrowing logic for {base_filename}. Durations: {initial_durations}. Skipping file.")
             continue # Skip this file for safety

        # Check if the list is empty (e.g., original lab file was empty)
        if not initial_phones:
             logging.warning(f"Skipping {base_filename} as no phones remained after processing (original lab might be empty or invalid).")
             continue
        # --- MODIFICATION END ---

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
            'midi_pitch_estimated': padded_midi_pitch, # Store padded MIDI pitch
            'phone_sequence': initial_phones,
            'duration_sequence': np.array(adjusted_durations, dtype=np.int32),
            'initial_duration_sequence': np.array(initial_durations, dtype=np.int32), # Keep for potential analysis/vis
            'voiced_mask': padded_voiced_mask, # Store padded voiced mask (as uint8)
            'unvoiced_flag': padded_unvoiced_flag, # Store padded unvoiced flag (as uint8)
            'original_unpadded_length': original_mel_frames, # Store the original length
            #'raw_audio': padded_raw_audio # Store padded/truncated raw audio
        }

        # Store unnormalized data for the *first* successfully processed file for visualization
        if first_file_data_for_vis is None:
             first_file_data_for_vis = {
                 'mel': pad_or_truncate(log_mel, target_frames, pad_value=0.0), # Pad original log_mel
                 'f0': pad_or_truncate(log_f0, target_frames, pad_value=0.0),   # Pad original log_f0
                 'midi_pitch': pad_or_truncate(midi_pitch, target_frames, pad_value=0), # Pad original midi_pitch
                 'phone_sequence': initial_phones,
                 'initial_duration_sequence': np.array(initial_durations, dtype=np.int32) # Use initial durations for vis
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

    # Check for duplicate base_filenames before saving
    filenames = [bf for bf in processed_data.keys()]
    duplicates = [item for item, count in collections.Counter(filenames).items() if count > 1]
    if duplicates:
        logging.warning(f"Duplicate base_filenames found in processed_data: {duplicates}")
        # Optionally, decide how to handle duplicates here - e.g., raise error, skip, merge?
        # For now, just log the warning.

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
            logging.info(f"Saved vocab_size ({vocab_size}) and normalization stats as HDF5 attributes.")

            # Save data for each file
            for base_filename, data in tqdm(processed_data.items(), desc="Saving to HDF5"):
                group = f.create_group(base_filename)
                # Convert phone sequence to IDs
                phone_sequence = data.get('phone_sequence') # Use .get for safety
                duration_sequence = data.get('duration_sequence') # Use .get for safety

                # --- Add Assertion & ID Conversion ---
                if phone_sequence is None or duration_sequence is None:
                    logging.error(f"Missing phone_sequence or duration_sequence for {base_filename}. Skipping.")
                    continue
                if len(phone_sequence) != len(duration_sequence):
                     logging.error(f"ASSERTION FAILED for {base_filename}: len(phone_sequence)={len(phone_sequence)} != len(duration_sequence)={len(duration_sequence)}")
                     # Skip this file if lengths don't match to avoid saving bad data
                     continue

                phone_sequence_ids = np.array([phone_to_id[p] for p in phone_sequence], dtype=np.int32)
                # Save the phone sequence IDs (phoneme-level)
                group.create_dataset('phone_sequence_ids', data=phone_sequence_ids)

                # --- Define mapping from internal keys to HDF5 dataset names ---
                internal_key_to_hdf5_name = {
                    'mel': data_config.get('mel_key', 'mel'),
                    'f0': data_config.get('f0_key', 'f0'),
                    'duration_sequence': data_config.get('duration_key', 'duration_sequence'),
                    'midi_pitch_estimated': data_config.get('midi_pitch_key', 'midi_pitch_estimated'),
                    #'raw_audio': data_config.get('raw_audio_key', 'raw_audio'),
                    'voiced_mask': 'voiced_mask', # Assuming internal key matches desired HDF5 name
                    'unvoiced_flag': 'unvoiced_flag', # Assuming internal key matches desired HDF5 name
                    'initial_duration_sequence': 'initial_duration_sequence', # Assuming internal key matches desired HDF5 name
                    'original_unpadded_length': 'original_unpadded_length', # Assuming internal key matches desired HDF5 name
                }

                # --- Save all other data items using mapped names ---
                for key, value in data.items():
                    # Skip phone_sequence as we saved phone_sequence_ids instead
                    if key == 'phone_sequence':
                        continue
                    try:
                        # Ensure value is suitable for saving
                        if isinstance(value, (np.ndarray, list, int, float)):
                            # Get the target HDF5 dataset name from the map
                            hdf5_dataset_name = internal_key_to_hdf5_name.get(key)
                            if hdf5_dataset_name: # Only save if key is in our map
                                group.create_dataset(hdf5_dataset_name, data=value)
                            # else: # Optional: Warn if a key in data isn't in our map
                            #    logging.warning(f"Key '{key}' found in processed data but not in internal_key_to_hdf5_name map for {base_filename}. Skipping save.")
                        else:
                            logging.warning(f"Skipping key '{key}' for {base_filename}: Unsupported data type {type(value)}")
                    except Exception as e:
                        logging.error(f"Error saving key '{key}' (mapped to '{hdf5_dataset_name}') for {base_filename}: {e}")

                # --- Calculate and Save Frame-Level Phone IDs (using the key from config) ---
                # This happens *after* the loop saving other keys from the data dict
                if 'duration_sequence' in data:
                    try:
                        # Use the duration sequence already saved in the HDF5 group by the loop above
                        # Or retrieve from data dict again if needed: duration_sequence_val = data['duration_sequence']
                        frame_level_phone_ids = np.repeat(phone_sequence_ids, data['duration_sequence'])
                        # Ensure final length is exactly target_frames
                        if len(frame_level_phone_ids) != target_frames:
                            logging.warning(f"Frame-level phone ID length mismatch for {base_filename}. Got {len(frame_level_phone_ids)}, expected {target_frames}. Clamping.")
                            frame_level_phone_ids = pad_or_truncate(frame_level_phone_ids, target_frames, pad_value=phone_to_id["<PAD>"])
                        # Save using the key specified in the config
                        group.create_dataset(data_config['phoneme_key'], data=frame_level_phone_ids, compression="gzip")
                    except Exception as e:
                        logging.error(f"Error creating/saving frame-level phone IDs ('{data_config['phoneme_key']}') for {base_filename}: {e}")
                else:
                     logging.error(f"Cannot create frame-level phone IDs for {base_filename} as 'duration_sequence' is missing from processed data.")
                # End of the main loop for base_filename, data in processed_data.items()

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
            mel=first_file_data_for_vis['mel'], # Unnormalized log-mel
            f0=first_file_data_for_vis['f0'],  # Unnormalized log-f0
            midi_pitch=first_file_data_for_vis['midi_pitch'], # Pass MIDI pitch
            phone_ids=example_phone_ids,
            durations=first_file_data_for_vis['initial_duration_sequence'], # Use initial durations key from the vis dict
            id_to_phone=id_to_phone,
            save_path=vis_save_path
        )
    else:
        logging.warning("Skipping visualization as no data was processed successfully.")

    logging.info("Preprocessing finished.")


if __name__ == "__main__":
    # Setup logging (can be called again, it's idempotent)
    # Consider adding command-line args for log level/file later
    setup_logging(level=logging.INFO)
    logger.info("Starting preprocessing...")
    # Example usage: Run preprocessing using the default config path
    try:
        preprocess_data()
        logger.info("Preprocessing completed successfully.")
    except Exception as e:
         logger.exception("Preprocessing failed with an error.") # Log full traceback