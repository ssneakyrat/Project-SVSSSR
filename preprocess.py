import os
import glob
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import logging

from utils.utils import (
    load_config, 
    extract_mel_spectrogram, 
    extract_f0, 
    normalize_mel_spectrogram, 
    expand_phone_seq_to_frames
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('preprocess')

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

def parse_midi_file(file_path):
    try:
        import mido
        
        midi_file = mido.MidiFile(file_path)
        
        # Extract note events
        notes = []
        current_time = 0
        
        for track in midi_file.tracks:
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append((current_time, msg.note, 'on'))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    notes.append((current_time, msg.note, 'off'))
        
        return notes
    except Exception as e:
        logger.error(f"Error parsing MIDI file {file_path}: {e}")
        return []

def preprocess_data(config, output_path):
    raw_dir = config['data']['raw_dir']
    
    # Find all lab files
    lab_files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    logger.info(f"Found {len(lab_files)} lab files")
    
    # Find all wav files
    wav_files = glob.glob(f"{raw_dir}/**/*.wav", recursive=True)
    logger.info(f"Found {len(wav_files)} wav files")
    
    # Find all midi files
    midi_files = glob.glob(f"{raw_dir}/**/*.mid", recursive=True)
    midi_files += glob.glob(f"{raw_dir}/**/*.midi", recursive=True)
    logger.info(f"Found {len(midi_files)} midi files")
    
    # Create mapping between filenames
    data_pairs = []
    for lab_file in lab_files:
        base_name = os.path.splitext(os.path.basename(lab_file))[0]
        
        # Find matching wav file
        matching_wav = [f for f in wav_files if os.path.splitext(os.path.basename(f))[0] == base_name]
        matching_midi = [f for f in midi_files if os.path.splitext(os.path.basename(f))[0] == base_name]
        
        if matching_wav and matching_midi:
            data_pairs.append({
                'lab': lab_file,
                'wav': matching_wav[0],
                'midi': matching_midi[0],
                'id': base_name
            })
    
    logger.info(f"Found {len(data_pairs)} complete data pairs (lab + wav + midi)")
    
    # Collect unique phonemes
    unique_phones = set()
    for item in tqdm(data_pairs, desc="Collecting unique phonemes"):
        phonemes = parse_lab_file(item['lab'])
        for _, _, phone in phonemes:
            unique_phones.add(phone)
    
    phone_map = sorted(list(unique_phones))
    phone_to_idx = {phone: idx for idx, phone in enumerate(phone_map)}
    
    logger.info(f"Found {len(phone_map)} unique phonemes")
    
    # Create H5 dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as h5f:
        # Store phoneme map
        dt = h5py.special_dtype(vlen=str)
        h5f.create_dataset('phone_map', data=np.array(phone_map, dtype=dt))
        
        # Calculate maximum expected lengths
        max_frames = int((config['audio']['max_audio_length'] * config['audio']['sample_rate']) / config['audio']['hop_length'])
        
        # Create datasets
        mel_specs = h5f.create_dataset('mel_spectrograms', 
                                       shape=(len(data_pairs), config['model']['mel_bins'], max_frames),
                                       dtype=np.float32)
        
        f0s = h5f.create_dataset('f0', 
                                 shape=(len(data_pairs), max_frames),
                                 dtype=np.float32)
        
        phone_labels = h5f.create_dataset('phone_label', 
                                          shape=(len(data_pairs), max_frames),
                                          dtype=np.int32)
        
        phone_durations = h5f.create_dataset('phone_duration', 
                                             shape=(len(data_pairs), len(phone_map)),
                                             dtype=np.int32)
        
        midi_labels = h5f.create_dataset('midi_label', 
                                         shape=(len(data_pairs), max_frames),
                                         dtype=np.int32)
        
        file_ids = h5f.create_dataset('file_ids', 
                                      shape=(len(data_pairs),),
                                      dtype=dt)
        
        # Process each data pair
        for idx, item in tqdm(enumerate(data_pairs), desc="Processing data", total=len(data_pairs)):
            # Extract and save mel spectrogram
            mel_spec = extract_mel_spectrogram(item['wav'], config)
            mel_spec = normalize_mel_spectrogram(mel_spec)
            
            # Ensure shape is correct (mel_bins, frames)
            if mel_spec.shape[0] == config['model']['mel_bins']:
                mel_spec_shaped = mel_spec[:, :max_frames]
            else:
                mel_spec_shaped = np.zeros((config['model']['mel_bins'], min(mel_spec.shape[1], max_frames)))
                mel_spec_shaped[:mel_spec.shape[0], :] = mel_spec[:, :max_frames]
            
            # Pad if necessary
            if mel_spec_shaped.shape[1] < max_frames:
                padded = np.zeros((config['model']['mel_bins'], max_frames))
                padded[:, :mel_spec_shaped.shape[1]] = mel_spec_shaped
                mel_spec_shaped = padded
            
            mel_specs[idx] = mel_spec_shaped
            
            # Extract and save F0
            f0 = extract_f0(item['wav'], config)
            
            # Ensure correct length
            if len(f0) < max_frames:
                padded_f0 = np.zeros(max_frames)
                padded_f0[:len(f0)] = f0
                f0 = padded_f0
            else:
                f0 = f0[:max_frames]
            
            f0s[idx] = f0
            
            # Process phoneme data
            phonemes = parse_lab_file(item['lab'])
            phone_seq = [phone_to_idx[p[2]] for p in phonemes]
            durations = [p[1] - p[0] for p in phonemes]
            
            # Normalize durations to frame-level (assuming time is in 10ms units)
            sr = config['audio']['sample_rate']
            hop_length = config['audio']['hop_length']
            durations_frames = [int((d / 10000) * (sr / hop_length)) for d in durations]
            
            # Ensure at least 1 frame per phoneme
            durations_frames = [max(1, d) for d in durations_frames]
            
            # Store phone sequence expanded to frame level
            frame_level_phones = expand_phone_seq_to_frames(phone_seq, durations_frames, max_frames)
            phone_labels[idx] = frame_level_phones
            
            # Store durations (padded if necessary)
            if len(durations_frames) < phone_durations.shape[1]:
                padded_durations = np.zeros(phone_durations.shape[1], dtype=np.int32)
                padded_durations[:len(durations_frames)] = durations_frames
                durations_frames = padded_durations
            else:
                durations_frames = durations_frames[:phone_durations.shape[1]]
            
            phone_durations[idx] = durations_frames
            
            # Process MIDI data
            midi_events = parse_midi_file(item['midi'])
            
            # Convert MIDI events to frame-level note representation
            frame_level_midi = np.zeros(max_frames, dtype=np.int32)
            
            current_note = 60  # Default to middle C
            for time, note, event_type in midi_events:
                frame_idx = int((time / 1000) * (sr / hop_length))
                if frame_idx >= max_frames:
                    break
                    
                if event_type == 'on':
                    current_note = note
                    
                # Fill from this point to next event or end
                frame_level_midi[frame_idx:] = current_note
            
            midi_labels[idx] = frame_level_midi
            
            # Store file ID
            file_ids[idx] = item['id']
    
    logger.info(f"Preprocessing complete. Data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for SVS')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Path for the output H5 file (overrides config)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    output_path = args.output
    if output_path is None:
        bin_dir = config['data']['bin_dir']
        bin_file = config['data']['bin_file']
        output_path = os.path.join(bin_dir, bin_file)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    preprocess_data(config, output_path)

if __name__ == "__main__":
    main()