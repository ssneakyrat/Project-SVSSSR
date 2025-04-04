import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile

from utils.utils import load_config
from models.svs_model import SVSModel
from data.dataset import SVSDataset, H5FileManager


def plot_spectrogram(mel_spectrogram, title="Generated Mel-Spectrogram", save_path=None):
    """
    Plot and optionally save mel spectrogram
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
        print(f"Spectrogram saved to {save_path}")
    
    plt.show()


def load_vocoder(vocoder_path):
    """
    Load pretrained vocoder model
    This is a placeholder. Replace with your actual vocoder loading logic.
    """
    try:
        # This is a placeholder. Replace with actual vocoder loading code.
        # For example, if using HiFi-GAN:
        # from hifigan import Generator
        # vocoder = Generator(...)
        # vocoder.load_state_dict(torch.load(vocoder_path)['generator'])
        print(f"Loaded vocoder from {vocoder_path}")
        return None  # Replace with actual vocoder
    except Exception as e:
        print(f"Error loading vocoder: {e}")
        return None


def generate_audio(mel_spectrogram, vocoder=None, save_path=None):
    """
    Generate audio from mel spectrogram using vocoder
    This is a placeholder. Replace with your actual audio generation logic.
    """
    if vocoder is None:
        print("No vocoder provided, cannot generate audio.")
        return None
    
    try:
        # This is a placeholder. Replace with actual vocoder code.
        # For example, if using HiFi-GAN:
        # mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0)
        # audio = vocoder(mel_tensor).squeeze().detach().cpu().numpy()
        
        # For now, we'll generate white noise with the expected length
        hop_length = 256
        audio_length = mel_spectrogram.shape[1] * hop_length
        audio = np.random.normal(0, 0.01, audio_length)
        
        if save_path:
            wavfile.write(save_path, 22050, audio)
            print(f"Audio saved to {save_path}")
        
        return audio
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Inference with SVS Latent Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/diffusion_model.yaml', help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--vocoder', type=str, default=None, help='Path to pretrained vocoder model')
    parser.add_argument('--h5_file', type=str, default=None, help='H5 file with test data')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to use from H5 file')
    parser.add_argument('--custom_input', type=str, default=None, help='Path to custom input data')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    try:
        model = SVSModel.load_from_checkpoint(args.checkpoint)
        model.eval()
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Model loaded from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load vocoder if provided
    vocoder = None
    if args.vocoder:
        vocoder = load_vocoder(args.vocoder)
    
    device = next(model.parameters()).device
    
    # Load test data
    if args.h5_file:
        # Load from H5 file
        try:
            dataset = SVSDataset(args.h5_file)
            sample = dataset[args.sample_idx]
            
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
            
            # Run inference
            with torch.no_grad():
                generated_mel = model.infer(f0, phone, duration, midi)
            
            # Convert to numpy
            generated_mel = generated_mel.squeeze().cpu().numpy()
            
            # Plot spectrograms
            plot_spectrogram(
                original_mel, 
                "Original Mel-Spectrogram",
                os.path.join(args.output_dir, "original_mel.png")
            )
            
            plot_spectrogram(
                generated_mel, 
                "Generated Mel-Spectrogram",
                os.path.join(args.output_dir, "generated_mel.png")
            )
            
            # Generate audio if vocoder is available
            if vocoder:
                original_audio = generate_audio(
                    original_mel, 
                    vocoder,
                    os.path.join(args.output_dir, "original.wav")
                )
                
                generated_audio = generate_audio(
                    generated_mel, 
                    vocoder,
                    os.path.join(args.output_dir, "generated.wav")
                )
            
        except Exception as e:
            print(f"Error processing H5 file: {e}")
            return
    
    elif args.custom_input:
        # Load custom input data
        # This is a placeholder. Implement loading logic for your custom inputs.
        print(f"Custom input from {args.custom_input} not implemented yet")
    
    else:
        print("No input data provided. Please specify --h5_file or --custom_input")
        return
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()