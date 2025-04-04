import os
import glob
import h5py
import numpy as np
import math
from tqdm import tqdm
import argparse
import logging

from utils.utils import (
    load_config, 
    extract_mel_spectrogram,
    extract_mel_spectrogram_variable_length,
    extract_f0, 
    normalize_mel_spectrogram, 
    pad_or_truncate_mel
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('preprocess')

def list_lab_files(raw_dir):
    if not os.path.exists(raw_dir):
        logger.error(f"Error: {raw_dir} directory not found!")
        return []
    
    files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    logger.info(f"Found {len(files)} .lab files in {raw_dir} directory")
    
    return files

def parse_lab_file(file_path):
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
    base_filename = os.path.splitext(os.path.basename(lab_file_path))[0]
    lab_dir = os.path.dirname(lab_file_path)
    
    wav_dir = lab_dir.replace('/lab/', '/wav/')
    if '/lab/' not in wav_dir:
        wav_dir = lab_dir.replace('\\lab\\', '\\wav\\')
    
    wav_file_path = os.path.join(wav_dir, f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    wav_file_path = os.path.join(raw_dir, "wav", f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    return None

def save_to_h5_variable_length(output_path, file_data, phone_map, config, data_key='mel_spectrograms'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mel_bins = config['model'].get('mel_bins', 80)
    
    max_audio_length = config['audio'].get('max_audio_length', 10.0)
    sample_rate = config['audio'].get('sample_rate', 22050)
    hop_length = config['audio'].get('hop_length', 256)
    max_time_frames = math.ceil(max_audio_length * sample_rate / hop_length)
    
    valid_items = 0
    max_length = 0
    for file_info in file_data.values():
        if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
            valid_items += 1
            curr_length = file_info['MEL_SPEC'].shape[1]
            max_length = max(max_length, curr_length)
    
    max_length = min(max_length, max_time_frames)
    logger.info(f"Maximum time frames: {max_length} (equivalent to {max_length * hop_length / sample_rate:.2f} seconds)")
    
    with h5py.File(output_path, 'w') as f:
        phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('phone_map', data=phone_map_array)
        
        dataset = f.create_dataset(
            data_key,
            shape=(valid_items, mel_bins, max_length),
            dtype=np.float32,
            chunks=(1, mel_bins, min(128, max_length))
        )
        
        lengths_dataset = f.create_dataset(
            'lengths',
            shape=(valid_items,),
            dtype=np.int32
        )
        
        file_ids = f.create_dataset(
            'file_ids',
            shape=(valid_items,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        f0s = f.create_dataset(
            'f0',
            shape=(valid_items, max_length),
            dtype=np.float32
        )
        
        phone_labels = f.create_dataset(
            'phone_label',
            shape=(valid_items, max_length),
            dtype=np.int32
        )
        
        dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        dataset.attrs['n_fft'] = config['audio']['n_fft']
        dataset.attrs['hop_length'] = config['audio']['hop_length']
        dataset.attrs['n_mels'] = config['audio']['n_mels']
        dataset.attrs['variable_length'] = True
        dataset.attrs['max_frames'] = max_length
        
        idx = 0
        with tqdm(total=len(file_data), desc="Saving to H5", unit="file") as pbar:
            for file_id, file_info in file_data.items():
                if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
                    mel_spec = file_info['MEL_SPEC']
                    mel_spec = normalize_mel_spectrogram(mel_spec)
                    
                    actual_length = min(mel_spec.shape[1], max_length)
                    lengths_dataset[idx] = actual_length
                    
                    if mel_spec.shape[1] > max_length:
                        mel_spec = mel_spec[:, :max_length]
                    elif mel_spec.shape[1] < max_length:
                        padded = np.zeros((mel_bins, max_length), dtype=np.float32)
                        padded[:, :mel_spec.shape[1]] = mel_spec
                        mel_spec = padded
                    
                    dataset[idx] = mel_spec
                    file_ids[idx] = file_id
                    
                    # Process F0
                    f0 = file_info['F0']
                    if f0 is not None:
                        if len(f0) > max_length:
                            f0 = f0[:max_length]
                        elif len(f0) < max_length:
                            padded_f0 = np.zeros(max_length, dtype=np.float32)
                            padded_f0[:len(f0)] = f0
                            f0 = padded_f0
                        f0s[idx] = f0
                    
                    # Process phone labels
                    phone_texts = file_info['PHONE_TEXT']
                    phone_starts = file_info['PHONE_START']
                    phone_ends = file_info['PHONE_END']
                    
                    phone_label = np.zeros(max_length, dtype=np.int32)
                    
                    # Map phonemes to corresponding frames
                    for i, (start, end, text) in enumerate(zip(phone_starts, phone_ends, phone_texts)):
                        # Convert time units to frame indices
                        start_frame = min(max_length-1, int(start * sample_rate / (hop_length * 10000)))
                        end_frame = min(max_length, int(end * sample_rate / (hop_length * 10000)))
                        
                        # Find phone index in phone_map
                        if text in phone_map:
                            phone_idx = phone_map.index(text)
                        else:
                            phone_idx = 0  # Default to first phone if not found
                        
                        # Assign the phone index to corresponding frames
                        phone_label[start_frame:end_frame] = phone_idx
                    
                    phone_labels[idx] = phone_label
                    
                    idx += 1
                
                pbar.update(1)
    
    logger.info(f"Saved {idx} mel spectrograms to {output_path} with variable length support")

def collect_unique_phonemes(lab_files):
    unique_phonemes = set()
    
    with tqdm(total=len(lab_files), desc="Collecting phonemes", unit="file") as pbar:
        for file_path in lab_files:
            phonemes = parse_lab_file(file_path)
            for _, _, phone in phonemes:
                unique_phonemes.add(phone)
            pbar.update(1)
    
    phone_map = sorted(list(unique_phonemes))
    logger.info(f"Collected {len(phone_map)} unique phonemes")
    
    return phone_map

def main():
    parser = argparse.ArgumentParser(description='Process lab files and save data to H5 file')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--raw_dir', type=str, help='Raw directory path (overrides config)')
    parser.add_argument('--output', type=str, help='Path for the output H5 file (overrides config)')
    parser.add_argument('--min_phonemes', type=int, default=5, help='Minimum phonemes required per file')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms', help='Key to use for data in the H5 file')
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length mel spectrograms')
    parser.add_argument('--max_audio_length', type=float, default=None, help='Maximum audio length in seconds')
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
    
    variable_length = args.variable_length
    if not variable_length and 'data' in config and 'variable_length' in config['data']:
        variable_length = config['data']['variable_length']
    
    max_audio_length = args.max_audio_length
    if max_audio_length is None and 'audio' in config and 'max_audio_length' in config['audio']:
        max_audio_length = config['audio']['max_audio_length']
    
    if max_audio_length is not None:
        config['audio']['max_audio_length'] = max_audio_length
    else:
        config['audio']['max_audio_length'] = 10.0
    
    lab_files = list_lab_files(raw_dir)
    
    phone_map = collect_unique_phonemes(lab_files)
    config['data']['phone_map'] = phone_map
    
    all_file_data = {}
    
    min_phoneme_count = args.min_phonemes
    skipped_files_count = 0
    processed_files_count = 0
    
    with tqdm(total=len(lab_files), desc="Processing files", unit="file") as pbar:
        for file_path in lab_files:
            phonemes = parse_lab_file(file_path)
            
            if len(phonemes) < min_phoneme_count:
                skipped_files_count += 1
                pbar.update(1)
                continue
            
            wav_file_path = find_wav_file(file_path, raw_dir)
            
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = base_filename
            
            mel_spec = None
            f0 = None
            if wav_file_path:
                if variable_length:
                    mel_spec = extract_mel_spectrogram_variable_length(wav_file_path, config)
                else:
                    mel_spec = extract_mel_spectrogram(wav_file_path, config)
                f0 = extract_f0(wav_file_path, config)
            
            phone_starts = np.array([p[0] for p in phonemes])
            phone_ends = np.array([p[1] for p in phonemes])
            phone_durations = phone_ends - phone_starts
            phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
            
            all_file_data[file_id] = {
                'PHONE_START': phone_starts,
                'PHONE_END': phone_ends,
                'PHONE_DURATION': phone_durations,
                'PHONE_TEXT': phone_texts,
                'FILE_NAME': np.array([file_path], dtype=h5py.special_dtype(vlen=str)),
                'MEL_SPEC': mel_spec,
                'F0': f0
            }
            
            processed_files_count += 1
            pbar.update(1)
    
    logger.info(f"Files processed: {processed_files_count}")
    logger.info(f"Files skipped: {skipped_files_count}")
    
    if all_file_data:
        if variable_length:
            save_to_h5_variable_length(output_path, all_file_data, phone_map, config, args.data_key)
        else:
            logger.error("Fixed-length output is not supported in this version. Use --variable_length flag.")
    else:
        logger.warning("No files were processed. H5 file was not created.")

if __name__ == "__main__":
    main()