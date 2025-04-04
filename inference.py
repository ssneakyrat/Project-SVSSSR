import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from models.svs_model import SVSModel
from models.base_model import BaseSVSModel
from data.dataset import SVSDataset, H5FileManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference with SVS Model')
    
    # Model configuration
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/svs_model.yaml', 
                        help='Path to configuration file')
    
    # Inference configuration
    parser.add_argument('--output_dir', type=str, default='outputs', 
                        help='Directory to save outputs')
    parser.add_argument('--vocoder', type=str, default=None, 
                        help='Path to vocoder model (optional)')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Sampling temperature (noise scale)')
    parser.add_argument('--num_samples', type=int, default=1, 
                        help='Number of samples to generate')
    
    # Input data configuration
    parser.add_argument('--h5_file', type=str, default=None, 
                        help='H5 file with test data')
    parser.add_argument('--sample_idx', type=int, default=0, 
                        help='Index of sample to use from H5 file')
    parser.add_argument('--custom_input', type=str, default=None, 
                        help='Path to custom input data (not implemented yet)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda or cpu)')
    
    return parser.parse_args()


def load_vocoder(vocoder_path: str) -> Optional[object]:
    """
    Load pretrained vocoder model (placeholder)
    
    Args:
        vocoder_path: Path to vocoder model
        
    Returns:
        Loaded vocoder model or None
    """
    try:
        logger.info(f"Loading vocoder from {vocoder_path}")
        # This is a placeholder. Replace with actual vocoder loading code.
        # For example, with HiFi-GAN:
        # from hifigan import Generator
        # vocoder = Generator(...)
        # vocoder.load_state_dict(torch.load(vocoder_path)['generator'])
        
        # For now, return None to indicate we're using placeholder audio
        return None
        
    except Exception as e:
        logger.error(f"Error loading vocoder: {e}")
        return None


def plot_spectrogram(
    mel_spectrogram: np.ndarray, 
    title: str = "Mel-Spectrogram", 
    save_path: Optional[str] = None
):
    """
    Plot and optionally save mel spectrogram
    
    Args:
        mel_spectrogram: Mel spectrogram array
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram,
        y_axis='mel',
        x_axis='time',
        sr=22050,
        hop_length=256
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Spectrogram saved to {save_path}")
    
    plt.close()


def generate_audio(
    mel_spectrogram: np.ndarray, 
    vocoder: Optional[object] = None, 
    sample_rate: int = 22050,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Generate audio from mel spectrogram using vocoder (placeholder)
    
    Args:
        mel_spectrogram: Mel spectrogram array
        vocoder: Vocoder model
        sample_rate: Audio sample rate
        save_path: Path to save the audio
        
    Returns:
        Generated audio waveform
    """
    if vocoder is None:
        logger.warning("No vocoder provided, generating placeholder audio")
        # Generate synthetic noise shaped like the spectrogram energy
        # This is a placeholder until a real vocoder is implemented
        
        hop_length = 256
        audio_length = mel_spectrogram.shape[1] * hop_length
        
        # Create noise shaped by the mel spectrogram energy
        # Sum across frequency to get energy per frame
        energy = np.mean(mel_spectrogram, axis=0)
        energy = np.repeat(energy, hop_length)
        
        # Generate noise shaped by the energy
        noise = np.random.normal(0, 0.01, audio_length)
        shaped_noise = noise * energy[:audio_length] * 0.1
        
        audio = shaped_noise
    else:
        # This is a placeholder. Replace with actual vocoder code.
        # For example, with HiFi-GAN:
        # mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0)
        # audio = vocoder(mel_tensor).squeeze().detach().cpu().numpy()
        
        # For now, we'll generate white noise as a placeholder
        hop_length = 256
        audio_length = mel_spectrogram.shape[1] * hop_length
        audio = np.random.normal(0, 0.01, audio_length)
    
    # Save audio if requested
    if save_path:
        # Normalize audio to avoid clipping
        audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
        wavfile.write(save_path, sample_rate, audio.astype(np.float32))
        logger.info(f"Audio saved to {save_path}")
    
    return audio


def denormalize_mel(
    mel: np.ndarray, 
    is_db_scale: bool = True
) -> np.ndarray:
    """
    Denormalize mel spectrogram from [0, 1] range back to original scale
    
    Args:
        mel: Normalized mel spectrogram
        is_db_scale: Whether the original scale was in dB
        
    Returns:
        Denormalized mel spectrogram
    """
    if mel is None:
        return None
        
    if is_db_scale:
        # For dB scale (normalized from -80 to 0)
        return mel * 80.0 - 80.0
    else:
        # For linear scale, this is just a placeholder
        # Actual denormalization may require min/max values from the original data
        return mel


def process_h5_sample(
    model: SVSModel,
    h5_file: str,
    sample_idx: int,
    device: str,
    output_dir: str,
    vocoder: Optional[object] = None,
    temperature: float = 1.0,
    num_samples: int = 1
):
    """
    Process a sample from an H5 file
    
    Args:
        model: SVS model
        h5_file: Path to H5 file
        sample_idx: Index of sample to use
        device: Device to run inference on
        output_dir: Directory to save outputs
        vocoder: Optional vocoder model
        temperature: Sampling temperature
        num_samples: Number of samples to generate
    """
    try:
        # Load test data
        dataset = SVSDataset(h5_file)
        sample = dataset[sample_idx]
        
        # Move tensors to device
        f0 = sample['f0'].to(device)
        phone = sample['phone'].to(device)
        duration = sample['duration'].to(device)
        midi = sample['midi'].to(device)
        
        # Add batch dimension
        f0 = f0.unsqueeze(0)
        phone = phone.unsqueeze(0)
        duration = duration.unsqueeze(0)
        midi = midi.unsqueeze(0)
        
        # Original mel for comparison
        original_mel = sample['mel'].squeeze().cpu().numpy()
        
        # Denormalize for visualization
        original_mel_denorm = denormalize_mel(original_mel)
        
        # Plot original mel spectrogram
        plot_spectrogram(
            original_mel_denorm, 
            "Original Mel-Spectrogram",
            os.path.join(output_dir, "original_mel.png")
        )
        
        # Generate audio from original mel if vocoder is available
        if vocoder:
            original_audio = generate_audio(
                original_mel_denorm, 
                vocoder,
                save_path=os.path.join(output_dir, "original.wav")
            )
        
        # Generate multiple samples
        for sample_num in range(num_samples):
            # Run inference
            with torch.no_grad():
                generated_mel = model.infer(f0, phone, duration, midi, temperature)
            
            # Convert to numpy
            generated_mel = generated_mel.squeeze().cpu().numpy()
            
            # Denormalize for visualization
            generated_mel_denorm = denormalize_mel(generated_mel)
            
            # Plot generated mel spectrogram
            plot_spectrogram(
                generated_mel_denorm, 
                f"Generated Mel-Spectrogram (Sample {sample_num+1})",
                os.path.join(output_dir, f"generated_mel_{sample_num}.png")
            )
            
            # Generate audio if vocoder is available
            if vocoder:
                generated_audio = generate_audio(
                    generated_mel_denorm, 
                    vocoder,
                    save_path=os.path.join(output_dir, f"generated_{sample_num}.wav")
                )
                
        logger.info(f"Successfully processed sample {sample_idx}")
        
    except Exception as e:
        logger.error(f"Error processing H5 file: {e}")
        return


def main():
    """Main inference function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load configuration
        config = BaseSVSModel.load_config(args.config)
        
        # Load model
        try:
            logger.info(f"Loading model from {args.checkpoint}")
            model = SVSModel.load_from_checkpoint(args.checkpoint, config=config)
            model.eval()
            model.to(args.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
        
        # Load vocoder if provided
        vocoder = None
        if args.vocoder:
            vocoder = load_vocoder(args.vocoder)
        
        # Run inference
        if args.h5_file:
            # Process sample from H5 file
            process_h5_sample(
                model=model,
                h5_file=args.h5_file,
                sample_idx=args.sample_idx,
                device=args.device,
                output_dir=args.output_dir,
                vocoder=vocoder,
                temperature=args.temperature,
                num_samples=args.num_samples
            )
        elif args.custom_input:
            # Process custom input (not implemented yet)
            logger.warning(f"Custom input from {args.custom_input} not implemented yet")
        else:
            logger.error("No input data provided. Please specify --h5_file or --custom_input")
            return
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during inference: {e}")
    finally:
        # Clean up resources
        H5FileManager.get_instance().close_all()


if __name__ == "__main__":
    main()