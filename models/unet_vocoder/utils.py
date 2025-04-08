import torch
import numpy as np
import librosa

def generate_noise(mel_spec, hop_length=256):
    """
    Generate random noise matching the temporal dimensions of the mel spectrogram
    
    Args:
        mel_spec: Mel spectrogram [B, T, M]
        hop_length: Number of audio samples per mel frame
        
    Returns:
        noise: Random noise [B, T*hop_length, 1]
    """
    batch_size, time_steps, _ = mel_spec.size()
    noise = torch.randn(batch_size, time_steps * hop_length, 1, device=mel_spec.device)
    return noise

def audio_to_mel(audio, config):
    """
    Convert audio waveform to mel spectrogram
    
    Args:
        audio: Audio waveform tensor [B, T]
        config: Configuration dictionary
        
    Returns:
        mel: Mel spectrogram [B, T//hop_length, n_mels]
    """
    # Extract audio parameters from config
    sample_rate = config['audio']['sample_rate']
    n_fft = config['audio']['n_fft']
    hop_length = config['audio']['hop_length']
    win_length = config['audio']['win_length']
    n_mels = config['audio']['n_mels']
    fmin = config['audio']['fmin']
    fmax = config['audio']['fmax']
    
    # Process each audio sample in the batch
    mel_batch = []
    for audio_sample in audio:
        # Convert to numpy for librosa
        audio_np = audio_sample.cpu().numpy()
        
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio_np, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, win_length=win_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Append to batch
        mel_batch.append(torch.from_numpy(log_mel.T).float())
    
    # Stack batch
    return torch.stack(mel_batch)

def normalize_mel(mel, mean, std):
    """
    Normalize mel spectrogram
    
    Args:
        mel: Mel spectrogram [B, T, M]
        mean: Mean values [M]
        std: Standard deviation values [M]
        
    Returns:
        normalized_mel: Normalized mel spectrogram [B, T, M]
    """
    return (mel - mean) / std