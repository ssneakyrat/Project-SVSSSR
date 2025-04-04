import yaml
import numpy as np
import librosa
import os
import torch

def load_config(config_path="config/model.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_mel_spectrogram(wav_path, config):
    y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
    
    # Check if audio exceeds maximum length
    max_audio_length = config['audio'].get('max_audio_length', 10.0)
    max_samples = int(max_audio_length * sr)
    
    if len(y) > max_samples:
        y = y[:max_samples]
    
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

def extract_f0(wav_path, config):
    y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
    
    # Check if audio exceeds maximum length
    max_audio_length = config['audio'].get('max_audio_length', 10.0)
    max_samples = int(max_audio_length * sr)
    
    if len(y) > max_samples:
        y = y[:max_samples]
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=config['audio']['f0_min'],
        fmax=config['audio']['f0_max'],
        sr=sr,
        hop_length=config['audio']['hop_length']
    )
    
    f0 = np.nan_to_num(f0)
    return f0

def normalize_mel_spectrogram(mel_spec):
    mel_spec = np.clip(mel_spec, -80.0, 0.0)
    mel_spec = (mel_spec + 80.0) / 80.0
    return mel_spec

def denormalize_mel_spectrogram(mel_spec):
    mel_spec = mel_spec * 80.0 - 80.0
    return mel_spec

def expand_phone_seq_to_frames(phone_seq, durations, total_frames):
    # Expand phoneme sequence to frame level based on durations
    frame_level_phones = []
    for phone, dur in zip(phone_seq, durations):
        frame_level_phones.extend([phone] * dur)
    
    # Truncate or pad if needed
    if len(frame_level_phones) > total_frames:
        frame_level_phones = frame_level_phones[:total_frames]
    elif len(frame_level_phones) < total_frames:
        # Pad last phone
        frame_level_phones.extend([frame_level_phones[-1]] * (total_frames - len(frame_level_phones)))
        
    return np.array(frame_level_phones)

def prepare_model_inputs(f0, phone_labels, phone_durations, midi_labels, device):
    """
    Prepare inputs for the model during inference
    """
    # Convert to tensors
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(0)
    phone_labels = torch.from_numpy(phone_labels).long().to(device).unsqueeze(0)
    phone_durations = torch.from_numpy(phone_durations).float().to(device).unsqueeze(0)
    midi_labels = torch.from_numpy(midi_labels).long().to(device).unsqueeze(0)
    
    return f0, phone_labels, phone_durations, midi_labels

def save_mel_plot(mel_spec, filename, title='Mel Spectrogram'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()