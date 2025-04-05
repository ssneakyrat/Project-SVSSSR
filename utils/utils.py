import yaml
import numpy as np
import librosa
import os
import math

def load_config(config_path="config/default.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using default configuration.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}

def extract_mel_spectrogram(wav_path, config):
    """Extract fixed-length mel spectrogram"""
    try:
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            win_length=config['audio']['win_length'],
            n_mels=config['audio']['n_mels'],
            fmin=config['audio']['fmin'],
            fmax=config['audio']['fmax'],
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec
        
    except Exception as e:
        print(f"Error extracting mel spectrogram from {wav_path}: {e}")
        return None

def extract_mel_spectrogram_variable_length(wav_path, config):
    """Extract variable-length mel spectrogram with maximum length constraint"""
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Check if audio exceeds maximum length
        max_audio_length = config['audio'].get('max_audio_length', 10.0)  # Default 10 seconds
        max_samples = int(max_audio_length * sr)
        
        if len(y) > max_samples:
            print(f"Warning: Audio file {wav_path} exceeds maximum length of {max_audio_length}s. Truncating.")
            y = y[:max_samples]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            win_length=config['audio']['win_length'],
            n_mels=config['audio']['n_mels'],
            fmin=config['audio']['fmin'],
            fmax=config['audio']['fmax'],
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure correct shape (freq_bins, time_frames)
        if mel_spec.shape[0] > mel_spec.shape[1]:
            mel_spec = mel_spec.T
            
        return mel_spec
        
    except Exception as e:
        print(f"Error extracting variable-length mel spectrogram from {wav_path}: {e}")
        return None

def extract_f0(wav_path, config):
    try:
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Check if audio exceeds maximum length
        max_audio_length = config['audio'].get('max_audio_length', 10.0)
        max_samples = int(max_audio_length * sr)
        
        if len(y) > max_samples:
            y = y[:max_samples]
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config.get('audio', {}).get('f0_min', 50),
            fmax=config.get('audio', {}).get('f0_max', 600),
            sr=sr,
            hop_length=config['audio']['hop_length']
        )
        
        f0 = np.nan_to_num(f0)
        return f0
        
    except Exception as e:
        print(f"Error extracting F0 from {wav_path}: {e}")
        return None

def normalize_mel_spectrogram(mel_spec):
    if np.min(mel_spec) < 0:
        mel_spec = np.clip(mel_spec, -80.0, 0.0)
        mel_spec = (mel_spec + 80.0) / 80.0
    else:
        max_val = np.max(mel_spec)
        min_val = np.min(mel_spec)
        if max_val > min_val:
            mel_spec = (mel_spec - min_val) / (max_val - min_val)
    
    return mel_spec

def pad_or_truncate_mel(mel_spec, target_length=128, target_bins=80):
    curr_bins, curr_length = mel_spec.shape
    
    resized_mel = np.zeros((target_bins, target_length), dtype=mel_spec.dtype)
    
    freq_bins_to_copy = min(curr_bins, target_bins)
    time_frames_to_copy = min(curr_length, target_length)
    
    resized_mel[:freq_bins_to_copy, :time_frames_to_copy] = mel_spec[:freq_bins_to_copy, :time_frames_to_copy]
    
    return resized_mel

def calculate_mel_frames_from_audio_length(audio_length_seconds, sample_rate, hop_length):
    """Calculate number of mel spectrogram frames from audio length in seconds"""
    # Calculate number of samples
    num_samples = int(audio_length_seconds * sample_rate)
    
    # Calculate number of frames
    # The formula accounts for the frame centered at each hop_length interval
    num_frames = 1 + (num_samples - 1) // hop_length
    
    return num_frames

def prepare_mel_for_model(mel_spec, target_length=None, target_bins=80, variable_length=False):
    """Prepare mel spectrogram for model input, with optional variable length support"""
    import torch
    
    # Make sure mel_spec is in the correct orientation (freq bins, time frames)
    # Typically mel_spec should have shape (freq_bins, time_frames)
    # If it's transposed, fix it
    if isinstance(mel_spec, np.ndarray):
        if mel_spec.shape[0] > mel_spec.shape[1]:  # If time > freq, transpose
            print(f"Warning: Transposing mel spectrogram from {mel_spec.shape} to {mel_spec.shape[::-1]}")
            mel_spec = mel_spec.T
    
    mel_spec = normalize_mel_spectrogram(mel_spec)
    
    if variable_length:
        # For variable length, we just ensure freq dimension matches target
        if mel_spec.shape[0] != target_bins:
            # Resize frequency dimension if needed
            resized = np.zeros((target_bins, mel_spec.shape[1]), dtype=mel_spec.dtype)
            freq_bins_to_copy = min(mel_spec.shape[0], target_bins)
            resized[:freq_bins_to_copy, :] = mel_spec[:freq_bins_to_copy, :]
            mel_spec = resized
    else:
        # For fixed length, we apply standard padding/truncation
        if target_length is not None:
            mel_spec = pad_or_truncate_mel(mel_spec, target_length, target_bins)
    
    # Convert to tensor
    mel_tensor = torch.from_numpy(mel_spec).float()
    
    # Add batch and channel dimensions: [freq, time] -> [1, 1, freq, time]
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)
    
    return mel_tensor