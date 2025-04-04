import os
import glob
import h5py
import numpy as np
import math
import argparse
import logging
from tqdm import tqdm

from utils.utils import (
    load_config, 
    extract_mel_spectrogram,
    extract_f0, 
    normalize_mel_spectrogram
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('preprocess')


def list_lab_files(raw_dir):
    """
    Find all lab files in the given directory (recursively)
    """
    if not os.path.exists(raw_dir):
        logger.error(f"Error: {raw_dir} directory not found!")
        return []
    
    files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    logger.info(f"Found {len(files)} .lab files in {raw_dir} directory")
    
    return files


def parse_lab_file(file_path):
    """
    Parse a lab file containing phoneme labels and timings
    Format: <start_time> <end_time> <phoneme>
    """
    phonemes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = int(parts[0])
                    end_time = int(parts[1])
                    phoneme = parts[2]
                    phonemes.append((start_time, end_time, phoneme))
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
    
    return phonemes


def find_wav_file(lab_file_path, raw_dir):
    """
    Find the corresponding wav file for a lab file
    """
    base_filename = os.path.splitext(os.path.basename(lab_file_path))[0]
    lab_dir = os.path.dirname(lab_file_path)
    
    # First try looking in the parallel wav directory
    wav_dir = lab_dir.replace('/lab/', '/wav/')
    if '/lab/' not in wav_dir:
        wav_dir = lab_dir.replace('\\lab\\', '\\wav\\')
    
    wav_file_path = os.path.join(wav_dir, f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    # Try looking in the raw_dir/wav directory
    wav_file_path = os.path.join(raw_dir, "wav", f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    return None


def find_midi_file(lab_file_path, raw_dir):
    """
    Find the corresponding MIDI file for a lab file
    """
    base_filename = os.path.splitext(os.path.basename(lab_file_path))[0]
    lab_dir = os.path.dirname(lab_file_path)
    
    # First try looking in the parallel midi directory
    midi_dir = lab_dir.replace('/lab/', '/midi/')
    if '/lab/' not in midi_dir:
        midi_dir = lab_dir.replace('\\lab\\', '\\midi\\')
    
    # Try different extensions
    for ext in ['.mid', '.midi']:
        midi_file_path = os.path.join(midi_dir, f"{base_filename}{ext}")
        
        if os.path.exists(midi_file_path):
            return midi_file_path
    
    # Try looking in the raw_dir/midi directory
    for ext in ['.mid', '.midi']:
        midi_file_path = os.path.join(raw_dir, "midi", f"{base_filename}{ext}")
        
        if os.path.exists(midi_file_path):
            return midi_file_path
    
    return None


def parse_midi_notes(midi_file, sample_rate, hop_length):
    """
    Extract MIDI notes from a MIDI file
    This is a placeholder. Implement proper MIDI parsing based on your data.
    """
    try:
        # This is just a placeholder. Replace with actual MIDI parsing.
        # For example, using pretty_midi:
        # import pretty_midi
        # midi_data = pretty_midi.PrettyMIDI(midi_file)
        # notes = []
        # for instrument in midi_data.instruments:
        #     for note in instrument.notes:
        #         start_frame = int(note.start * sample_rate / hop_length)
        #         end_frame = int(note.end * sample_rate / hop_length)
        #         notes.append((start_frame, end_frame, note.pitch))
        
        # For now, return a default C4 (MIDI note 60)
        # This would need to be replaced with real parsing
        notes = [(0, 1000, 60)]  # Placeholder
        return notes
    except Exception as e:
        logger.error(f"Error parsing MIDI file {midi_file}: {e}")
        return []


def convert_lab_to_frame_indices(phonemes, sample_rate, hop_length):
    """
    Convert phoneme timings to frame indices
    """
    frame_phonemes = []
    for start_time, end_time, phoneme in phonemes:
        # Convert time in UASR time units to seconds
        # This depends on your lab file format - adjust as needed
        start_sec = start_time / 10000000
        end_sec = end_time / 10000000
        
        # Convert seconds to frame indices
        start_frame = int(start_sec * sample_rate / hop_length)
        end_frame = int(end_sec * sample_rate / hop_length)
        
        frame_phonemes.append((start_frame, end_frame, phoneme))
    
    return frame_phonemes


def create_frame_level_data(phonemes, midi_notes, num_frames, phone_map):
    """
    Create frame-level phoneme, duration, and MIDI data
    """
    # Initialize arrays
    phone_ids = np.zeros(num_frames, dtype=np.int32)
    durations = np.zeros(num_frames, dtype=np.float32)
    midi_values = np.zeros(num_frames, dtype=np.int32)
    
    # Fill in phoneme and duration data
    for start_frame, end_frame, phoneme in phonemes:
        if end_frame > start_frame:
            if phoneme in phone_map:
                phone_id = phone_map.index(phoneme)
            else:
                # Use a special ID for unknown phones
                phone_id = 0
                
            # Limit to valid frame range
            start_frame = max(0, min(start_frame, num_frames-1))
            end_frame = max(0, min(end_frame, num_frames))
            
            # Set phone ID for all frames in this segment
            phone_ids[start_frame:end_frame] = phone_id
            
            # Calculate duration in seconds and set for all frames
            duration_sec = (end_frame - start_frame) / (num_frames / 5.0)  # Assuming 5 seconds total
            durations[start_frame:end_frame] = duration_sec
    
    # Fill in MIDI note data
    for start_frame, end_frame, pitch in midi_notes:
        # Limit to valid frame range
        start_frame = max(0, min(start_frame, num_frames-1))
        end_frame = max(0, min(end_frame, num_frames))
        
        # Set MIDI pitch for all frames in this segment
        midi_values[start_frame:end_frame] = pitch
    
    return phone_ids, durations, midi_values


def collect_unique_phonemes(lab_files):
    """
    Collect all unique phonemes from the lab files
    """
    unique_phonemes = set()
    
    with tqdm(total=len(lab_files), desc="Collecting phonemes", unit="file") as pbar:
        for file_path in lab_files:
            phonemes = parse_lab_file(file_path)
            for _, _, phone in phonemes:
                unique_phonemes.add(phone)
            pbar.update(1)
    
    # Always include padding and unknown tokens
    unique_phonemes.add("<pad>")
    unique_phonemes.add("<unk>")
    
    phone_map = sorted(list(unique_phonemes))
    logger.info(f"Collected {len(phone_map)} unique phonemes")
    
    return phone_map


def save_to_h5(output_path, file_data, phone_map, config, data_key='mel_spectrograms'):
    """
    Save processed data to H5 file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mel_bins = config['model'].get('mel_bins', 80)
    time_frames = config['model'].get('time_frames', 432)  # Default ~5 seconds at 22050Hz/256 hop
    
    # Count valid items
    valid_items = sum(1 for file_info in file_data.values() 
                       if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None)
    
    with h5py.File(output_path, 'w') as f:
        # Store phone map
        phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('phone_map', data=phone_map_array)
        
        # Create datasets
        mel_dataset = f.create_dataset(
            data_key,
            shape=(valid_items, mel_bins, time_frames),
            dtype=np.float32,
            chunks=(1, mel_bins, time_frames)
        )
        
        f0_dataset = f.create_dataset(
            'f0',
            shape=(valid_items, time_frames),
            dtype=np.float32,
            chunks=(1, time_frames)
        )
        
        phone_dataset = f.create_dataset(
            'phone',
            shape=(valid_items, time_frames),
            dtype=np.int32,
            chunks=(1, time_frames)
        )
        
        duration_dataset = f.create_dataset(
            'duration',
            shape=(valid_items, time_frames),
            dtype=np.float32,
            chunks=(1, time_frames)
        )
        
        midi_dataset = f.create_dataset(
            'midi',
            shape=(valid_items, time_frames),
            dtype=np.int32,
            chunks=(1, time_frames)
        )
        
        file_ids = f.create_dataset(
            'file_ids',
            shape=(valid_items,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Store audio parameters as attributes
        mel_dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        mel_dataset.attrs['n_fft'] = config['audio']['n_fft']
        mel_dataset.attrs['hop_length'] = config['audio']['hop_length']
        mel_dataset.attrs['n_mels'] = config['audio']['n_mels']
        
        # Store data
        idx = 0
        with tqdm(total=len(file_data), desc="Saving to H5", unit="file") as pbar:
            for file_id, file_info in file_data.items():
                if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
                    # Process mel spectrogram
                    mel_spec = file_info['MEL_SPEC']
                    mel_spec = normalize_mel_spectrogram(mel_spec)
                    
                    # Pad or truncate to target length
                    if mel_spec.shape[1] > time_frames:
                        mel_spec = mel_spec[:, :time_frames]
                    elif mel_spec.shape[1] < time_frames:
                        padded = np.zeros((mel_bins, time_frames), dtype=np.float32)
                        padded[:, :mel_spec.shape[1]] = mel_spec
                        mel_spec = padded
                    
                    # Store data
                    mel_dataset[idx] = mel_spec
                    f0_dataset[idx] = file_info['F0'][:time_frames] if 'F0' in file_info else np.zeros(time_frames)
                    phone_dataset[idx] = file_info['PHONE_IDS'][:time_frames] if 'PHONE_IDS' in file_info else np.zeros(time_frames)
                    duration_dataset[idx] = file_info['DURATIONS'][:time_frames] if 'DURATIONS' in file_info else np.zeros(time_frames)
                    midi_dataset[idx] = file_info['MIDI'][:time_frames] if 'MIDI' in file_info else np.zeros(time_frames)
                    file_ids[idx] = file_id
                    
                    idx += 1
                
                pbar.update(1)
    
    logger.info(f"Saved {idx} items to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess audio and label data for SVS training')
    parser.add_argument('--config', type=str, default='config/svs_model.yaml', help='Path to configuration file')
    parser.add_argument('--raw_dir', type=str, help='Raw directory path (overrides config)')
    parser.add_argument('--output', type=str, help='Path for the output H5 file (overrides config)')
    parser.add_argument('--min_phonemes', type=int, default=5, help='Minimum phonemes required per file')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms', help='Key to use for data in the H5 file')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.raw_dir:
        config['data']['raw_dir'] = args.raw_dir
    
    raw_dir = config['data']['raw_dir']
    
    output_path = args.output
    if output_path is None:
        bin_dir = config['data']['bin_dir']
        bin_file = config['data']['bin_file']
        output_path = os.path.join(bin_dir, bin_file)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Find all lab files
    lab_files = list_lab_files(raw_dir)
    
    # Apply max_samples limit if specified
    if args.max_samples and args.max_samples > 0:
        lab_files = lab_files[:args.max_samples]
    
    # Collect unique phonemes
    phone_map = collect_unique_phonemes(lab_files)
    
    # Make sure special tokens are at the beginning
    if "<pad>" in phone_map:
        phone_map.remove("<pad>")
    if "<unk>" in phone_map:
        phone_map.remove("<unk>")
    
    phone_map = ["<pad>", "<unk>"] + phone_map
    
    # Store phone map in config
    config['data']['phone_map'] = phone_map
    
    # Process files
    all_file_data = {}
    
    min_phoneme_count = args.min_phonemes
    skipped_files_count = 0
    processed_files_count = 0
    
    sample_rate = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    
    with tqdm(total=len(lab_files), desc="Processing files", unit="file") as pbar:
        for file_path in lab_files:
            # Parse lab file
            phonemes = parse_lab_file(file_path)
            
            # Skip files with too few phonemes
            if len(phonemes) < min_phoneme_count:
                skipped_files_count += 1
                pbar.update(1)
                continue
            
            # Find corresponding wav file
            wav_file_path = find_wav_file(file_path, raw_dir)
            
            # Skip files without wav files
            if not wav_file_path:
                skipped_files_count += 1
                pbar.update(1)
                continue
            
            # Find corresponding MIDI file
            midi_file_path = find_midi_file(file_path, raw_dir)
            
            # Get file ID
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = base_filename
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(wav_file_path, config)
            
            # Skip files with failed mel extraction
            if mel_spec is None:
                skipped_files_count += 1
                pbar.update(1)
                continue
            
            # Extract F0
            f0 = extract_f0(wav_file_path, config)
            
            # Convert lab timings to frame indices
            frame_phonemes = convert_lab_to_frame_indices(phonemes, sample_rate, hop_length)
            
            # Parse MIDI notes if available
            midi_notes = []
            if midi_file_path:
                midi_notes = parse_midi_notes(midi_file_path, sample_rate, hop_length)
            
            # If no MIDI file, use a placeholder C4 (MIDI note 60)
            if not midi_notes:
                midi_notes = [(0, mel_spec.shape[1], 60)]
            
            # Create frame-level data
            phone_ids, durations, midi_values = create_frame_level_data(
                frame_phonemes, midi_notes, mel_spec.shape[1], phone_map
            )
            
            # Store data
            all_file_data[file_id] = {
                'MEL_SPEC': mel_spec,
                'F0': f0 if f0 is not None else np.zeros(mel_spec.shape[1]),
                'PHONE_IDS': phone_ids,
                'DURATIONS': durations,
                'MIDI': midi_values,
                'FILE_NAME': file_path
            }
            
            processed_files_count += 1
            pbar.update(1)
    
    logger.info(f"Files processed: {processed_files_count}")
    logger.info(f"Files skipped: {skipped_files_count}")
    
    # Save data to H5 file
    if all_file_data:
        save_to_h5(output_path, all_file_data, phone_map, config, args.data_key)
    else:
        logger.warning("No files were processed. H5 file was not created.")


if __name__ == "__main__":
    main()