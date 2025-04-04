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

def extract_mel_spectrogram_variable_length(wav_path, config):
    y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
    
    max_audio_length = config['audio'].get('max_audio_length', 10.0)
    max_samples = int(max_audio_length * sr)
    
    if len(y) > max_samples:
        y = y[:max_samples]
    
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        win_length=config['audio'].get('win_length', config['audio']['n_fft']),
        n_mels=config['audio']['n_mels'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax'],
    )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

def extract_f0(wav_path, config):
    y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
    
    max_audio_length = config['audio'].get('max_audio_length', 10.0)
    max_samples = int(max_audio_length * sr)
    
    if len(y) > max_samples:
        y = y[:max_samples]
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=config['audio'].get('f0_min', 50),
        fmax=config['audio'].get('f0_max', 600),
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

def pad_or_truncate_mel(mel_spec, target_shape):
    mel_bins, time_frames = target_shape
    current_bins, current_frames = mel_spec.shape
    
    if current_bins == mel_bins and current_frames == time_frames:
        return mel_spec
    
    padded_mel = np.zeros(target_shape, dtype=np.float32)
    
    bins_to_copy = min(current_bins, mel_bins)
    frames_to_copy = min(current_frames, time_frames)
    
    padded_mel[:bins_to_copy, :frames_to_copy] = mel_spec[:bins_to_copy, :frames_to_copy]
    
    return padded_mel

def expand_phone_seq_to_frames(phone_seq, durations, total_frames):
    frame_level_phones = []
    for phone, dur in zip(phone_seq, durations):
        frame_level_phones.extend([phone] * dur)
    
    if len(frame_level_phones) > total_frames:
        frame_level_phones = frame_level_phones[:total_frames]
    elif len(frame_level_phones) < total_frames:
        if frame_level_phones:
            frame_level_phones.extend([frame_level_phones[-1]] * (total_frames - len(frame_level_phones)))
        else:
            frame_level_phones = [0] * total_frames
        
    return np.array(frame_level_phones)

def prepare_model_inputs(f0, phone_labels, phone_durations, midi_labels, device):
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(0)
    phone_labels = torch.from_numpy(phone_labels).long().to(device).unsqueeze(0)
    phone_durations = torch.from_numpy(phone_durations).float().to(device).unsqueeze(0)
    midi_labels = torch.from_numpy(midi_labels).long().to(device).unsqueeze(0)
    
    return f0, phone_labels, phone_durations, midi_labels