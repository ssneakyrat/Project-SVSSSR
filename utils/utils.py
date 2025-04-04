import yaml
import numpy as np
import librosa
import os
import math
import torch


def load_config(config_path="config/default.yaml"):
    """
    Load configuration from YAML file
    """
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
    """
    Extract mel spectrogram from audio file
    """
    try:
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Check if audio exceeds maximum length
        max_audio_length = config['audio'].get('max_audio_length', 5.0)
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
        
        # Convert to dB scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
        
    except Exception as e:
        print(f"Error extracting mel spectrogram from {wav_path}: {e}")
        return None


def extract_f0(wav_path, config):
    """
    Extract F0 contour from audio file
    """
    try:
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Check if audio exceeds maximum length
        max_audio_length = config['audio'].get('max_audio_length', 5.0)
        max_samples = int(max_audio_length * sr)
        
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Extract F0 using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config['audio']['f0_min'],
            fmax=config['audio']['f0_max'],
            sr=sr,
            hop_length=config['audio']['hop_length']
        )
        
        # Replace NaN values with zeros
        f0 = np.nan_to_num(f0)
        
        return f0
        
    except Exception as e:
        print(f"Error extracting F0 from {wav_path}: {e}")
        return None


def normalize_mel_spectrogram(mel_spec):
    """
    Normalize mel spectrogram to [0, 1] range
    """
    if mel_spec is None:
        return None
        
    if np.min(mel_spec) < 0:
        # For dB scale mel spectrograms (typically -80 to 0)
        mel_spec = np.clip(mel_spec, -80.0, 0.0)
        mel_spec = (mel_spec + 80.0) / 80.0
    else:
        # For power scale mel spectrograms
        max_val = np.max(mel_spec)
        min_val = np.min(mel_spec)
        if max_val > min_val:
            mel_spec = (mel_spec - min_val) / (max_val - min_val)
    
    return mel_spec


def denormalize_mel_spectrogram(mel_spec, is_db_scale=True):
    """
    Denormalize mel spectrogram from [0, 1] range back to original scale
    """
    if mel_spec is None:
        return None
        
    if is_db_scale:
        # For dB scale (normalized from -80 to 0)
        return mel_spec * 80.0 - 80.0
    else:
        # For linear scale, denormalization may need min/max values from the original data
        # This is just a placeholder assuming 0 to 1 range
        return mel_spec


def pad_or_truncate_sequence(sequence, target_length):
    """
    Pad or truncate a sequence to target length
    """
    current_length = len(sequence)
    
    if current_length > target_length:
        # Truncate
        return sequence[:target_length]
    elif current_length < target_length:
        # Pad with zeros
        if isinstance(sequence, np.ndarray):
            padding = np.zeros((target_length - current_length,) + sequence.shape[1:])
            return np.concatenate([sequence, padding], axis=0)
        elif isinstance(sequence, torch.Tensor):
            padding = torch.zeros((target_length - current_length,) + tuple(sequence.shape[1:]), 
                                  dtype=sequence.dtype, device=sequence.device)
            return torch.cat([sequence, padding], dim=0)
        else:
            raise TypeError(f"Unsupported sequence type: {type(sequence)}")
    else:
        # No change needed
        return sequence


def calculate_model_size(model):
    """
    Calculate the number of parameters and size in MB of a PyTorch model
    """
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        'parameters': num_params,
        'size_mb': size_mb
    }


def prepare_inference_batch(f0, phone, duration, midi, device=None):
    """
    Prepare input data for model inference
    """
    # Convert inputs to tensors if they're not already
    if not isinstance(f0, torch.Tensor):
        f0 = torch.tensor(f0, dtype=torch.float32)
    
    if not isinstance(phone, torch.Tensor):
        phone = torch.tensor(phone, dtype=torch.long)
    
    if not isinstance(duration, torch.Tensor):
        duration = torch.tensor(duration, dtype=torch.float32)
    
    if not isinstance(midi, torch.Tensor):
        midi = torch.tensor(midi, dtype=torch.long)
    
    # Add batch dimension if needed
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
    
    if phone.dim() == 1:
        phone = phone.unsqueeze(0)
    
    if duration.dim() == 1:
        duration = duration.unsqueeze(0)
    
    if midi.dim() == 1:
        midi = midi.unsqueeze(0)
    
    # Move to device if specified
    if device is not None:
        f0 = f0.to(device)
        phone = phone.to(device)
        duration = duration.to(device)
        midi = midi.to(device)
    
    return f0, phone, duration, midi