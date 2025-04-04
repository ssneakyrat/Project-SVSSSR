import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io
from PIL import Image
import logging
from typing import Dict, Any, Tuple, Optional, List, Union

logger = logging.getLogger(__name__)


def audio_to_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: int = 0,
    fmax: int = 8000
) -> np.ndarray:
    """
    Convert audio waveform to mel spectrogram
    
    Args:
        y: Audio waveform
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        win_length: Window length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel spectrogram
    """
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    
    # Convert to dB scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec


def normalize_mel(
    mel_spec: np.ndarray, 
    min_level_db: float = -80.0
) -> np.ndarray:
    """
    Normalize mel spectrogram to [0, 1] range
    
    Args:
        mel_spec: Mel spectrogram
        min_level_db: Minimum dB level
        
    Returns:
        Normalized mel spectrogram
    """
    if mel_spec.min() < 0:
        # For dB scale (typically -80 to 0)
        mel_spec = np.clip(mel_spec, min_level_db, 0.0)
        mel_spec = (mel_spec - min_level_db) / (-min_level_db)
    else:
        # For linear scale
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-7)
    
    return mel_spec


def denormalize_mel(
    norm_mel: np.ndarray, 
    min_level_db: float = -80.0
) -> np.ndarray:
    """
    Denormalize mel spectrogram from [0, 1] back to dB scale
    
    Args:
        norm_mel: Normalized mel spectrogram
        min_level_db: Minimum dB level
        
    Returns:
        Denormalized mel spectrogram in dB scale
    """
    return norm_mel * (-min_level_db) + min_level_db


def extract_f0_from_audio(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    f0_min: int = 50,
    f0_max: int = 600
) -> np.ndarray:
    """
    Extract F0 contour from audio
    
    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length between frames
        f0_min: Minimum fundamental frequency
        f0_max: Maximum fundamental frequency
        
    Returns:
        F0 contour array
    """
    # Extract F0 using PYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        hop_length=hop_length
    )
    
    # Replace NaN values with zeros
    f0 = np.nan_to_num(f0)
    
    return f0


def mel_to_image(
    mel: np.ndarray, 
    title: str = "Mel Spectrogram"
) -> np.ndarray:
    """
    Convert mel spectrogram to an image for visualization
    
    Args:
        mel: Mel spectrogram
        title: Plot title
        
    Returns:
        Image as numpy array
    """
    # Use matplotlib to create a spectrogram visualization
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel,
        y_axis='mel',
        x_axis='time',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close()
    
    return img


def create_comparison_plot(
    original_mel: np.ndarray, 
    generated_mel: np.ndarray, 
    title: str = "Comparison", 
    save_path: Optional[str] = None
):
    """
    Create a comparison plot of original and generated mel spectrograms
    
    Args:
        original_mel: Original mel spectrogram
        generated_mel: Generated mel spectrogram
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        original_mel, 
        y_axis='mel', 
        x_axis='time',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Original: {title}")
    
    # Plot generated
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        generated_mel, 
        y_axis='mel', 
        x_axis='time',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Generated: {title}")
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.close()


def save_audio(
    audio: np.ndarray, 
    path: str, 
    sample_rate: int = 22050
):
    """
    Save audio to file
    
    Args:
        audio: Audio waveform
        path: Path to save the audio
        sample_rate: Sample rate
    """
    # Normalize audio to avoid clipping
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Save as WAV file
    wavfile.write(path, sample_rate, audio.astype(np.float32))
    logger.info(f"Audio saved to {path}")


class DummyVocoder:
    """
    Dummy vocoder for testing when no proper vocoder is available.
    Generates audio shaped by the energy of the mel spectrogram.
    """
    def __init__(
        self, 
        hop_length: int = 256, 
        sample_rate: int = 22050
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
    
    def __call__(
        self, 
        mel: np.ndarray
    ) -> np.ndarray:
        """
        Generate audio from mel spectrogram
        
        Args:
            mel: Mel spectrogram [freq, time]
            
        Returns:
            Audio waveform
        """
        # Get approximate energy from mel spectrogram
        # Sum across frequency axis to get energy per frame
        energy = np.mean(mel, axis=0)
        
        # Repeat energy to match audio length
        audio_length = mel.shape[1] * self.hop_length
        energy = np.repeat(energy, self.hop_length)
        
        # Pad or truncate to match audio length
        if len(energy) < audio_length:
            energy = np.pad(energy, (0, audio_length - len(energy)))
        else:
            energy = energy[:audio_length]
        
        # Generate noise shaped by the energy
        noise = np.random.normal(0, 0.01, audio_length)
        shaped_noise = noise * energy * 0.1
        
        return shaped_noise


def visualize_attention(
    attention_matrix: np.ndarray,
    title: str = "Attention Matrix",
    save_path: Optional[str] = None
):
    """
    Visualize an attention matrix
    
    Args:
        attention_matrix: Attention matrix
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Target (Decoder)")
    plt.ylabel("Source (Encoder)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Attention plot saved to {save_path}")
    
    plt.close()


def visualize_model_output(
    output_dict: Dict[str, np.ndarray], 
    save_dir: str, 
    prefix: str = "model_output"
):
    """
    Visualize model outputs including mel spectrograms, attention, etc.
    
    Args:
        output_dict: Dictionary of output arrays
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize mel spectrograms
    if 'mel_original' in output_dict and 'mel_generated' in output_dict:
        create_comparison_plot(
            output_dict['mel_original'],
            output_dict['mel_generated'],
            "Mel Spectrogram",
            os.path.join(save_dir, f"{prefix}_mel_comparison.png")
        )
    
    # Visualize attention if available
    if 'attention' in output_dict:
        visualize_attention(
            output_dict['attention'],
            "Attention Matrix",
            os.path.join(save_dir, f"{prefix}_attention.png")
        )
    
    # Save audio if available
    if 'audio' in output_dict:
        save_audio(
            output_dict['audio'],
            os.path.join(save_dir, f"{prefix}_audio.wav")
        )


def plot_training_curve(
    losses: List[float], 
    title: str = "Training Loss", 
    save_path: Optional[str] = None
):
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Loss plot saved to {save_path}")
    
    plt.close()