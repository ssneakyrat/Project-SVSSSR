import os
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from models.attention_free_svs import AttentionFreeSVS
from utils.utils import load_config

def load_model(model_path, config):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to the checkpoint file
        config: Model configuration
        
    Returns:
        Loaded model
    """
    model = AttentionFreeSVS.load_from_checkpoint(
        model_path,
        config=config
    )
    model.eval()
    return model

def load_vocoder(vocoder_path, device):
    """
    Load a pretrained vocoder
    
    Args:
        vocoder_path: Path to the vocoder checkpoint
        device: Device to load the vocoder on
        
    Returns:
        Loaded vocoder model
    """
    # This is a placeholder. In a real implementation, you would load the actual
    # vocoder (e.g., HiFiGAN, MelGAN, etc.) here.
    print(f"Loading vocoder from {vocoder_path}")
    
    # Return dummy vocoder for demonstration
    class DummyVocoder:
        def __init__(self):
            self.device = device
            self.is_loaded = True
        
        def generate(self, mel):
            # This would normally convert mel spectrogram to audio
            # For now, just return random noise as placeholder
            time_frames = mel.shape[-1]
            sample_rate = 22050
            hop_length = 256
            audio_length = time_frames * hop_length
            return torch.randn(1, audio_length, device=self.device)
    
    return DummyVocoder()

def prepare_input_data(phone_path, f0_path, duration_path, midi_path, device):
    """
    Load and prepare input data for inference
    
    Args:
        phone_path: Path to phoneme sequence file
        f0_path: Path to F0 contour file
        duration_path: Path to duration file
        midi_path: Path to MIDI sequence file
        device: Device to load the data on
        
    Returns:
        Tuple of (phone_ids, f0, durations, midi_ids)
    """
    # In a real implementation, these would load actual data
    # For now, create dummy data
    
    # Load phoneme sequence (text format: one phoneme ID per line)
    if os.path.exists(phone_path):
        with open(phone_path, 'r') as f:
            phone_ids = [int(line.strip()) for line in f.readlines()]
        phone_ids = torch.tensor(phone_ids, dtype=torch.long, device=device)
    else:
        # Dummy data: 100 time steps with random phoneme IDs
        phone_ids = torch.randint(0, 50, (100,), dtype=torch.long, device=device)
    
    # Load F0 contour (text format: one F0 value per line)
    if os.path.exists(f0_path):
        with open(f0_path, 'r') as f:
            f0_values = [float(line.strip()) for line in f.readlines()]
        f0 = torch.tensor(f0_values, dtype=torch.float, device=device).unsqueeze(-1)
    else:
        # Dummy data: 100 time steps with random F0 values (100-600 Hz)
        f0 = torch.rand(100, 1, device=device) * 500 + 100
    
    # Load durations (text format: one duration value per line)
    if os.path.exists(duration_path):
        with open(duration_path, 'r') as f:
            duration_values = [float(line.strip()) for line in f.readlines()]
        durations = torch.tensor(duration_values, dtype=torch.float, device=device).unsqueeze(-1)
    else:
        # Dummy data: 100 time steps with random durations (1-10 frames)
        durations = torch.randint(1, 10, (100, 1), dtype=torch.float, device=device)
    
    # Load MIDI sequence (text format: one MIDI note ID per line)
    if os.path.exists(midi_path):
        with open(midi_path, 'r') as f:
            midi_ids = [int(line.strip()) for line in f.readlines()]
        midi_ids = torch.tensor(midi_ids, dtype=torch.long, device=device)
    else:
        # Dummy data: 100 time steps with random MIDI note IDs (36-84)
        midi_ids = torch.randint(36, 84, (100,), dtype=torch.long, device=device)
    
    # Ensure all inputs have the same length
    min_length = min(phone_ids.shape[0], f0.shape[0], durations.shape[0], midi_ids.shape[0])
    
    phone_ids = phone_ids[:min_length]
    f0 = f0[:min_length]
    durations = durations[:min_length]
    midi_ids = midi_ids[:min_length]
    
    # Add batch dimension
    phone_ids = phone_ids.unsqueeze(0)
    f0 = f0.unsqueeze(0)
    durations = durations.unsqueeze(0)
    midi_ids = midi_ids.unsqueeze(0)
    
    return phone_ids, f0, durations, midi_ids

def save_mel_spectrogram(mel, output_path):
    """
    Save a mel spectrogram to a file
    
    Args:
        mel: Mel spectrogram tensor [channels, mel_bins, time_frames]
        output_path: Path to save the mel spectrogram to
    """
    # Remove batch and channel dimensions
    if mel.dim() > 2:
        mel = mel.squeeze(0).squeeze(0)
    
    # Convert to numpy
    mel_np = mel.cpu().numpy()
    
    # Save as .npy file
    np.save(output_path, mel_np)
    print(f"Saved mel spectrogram to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate mel spectrograms with Attention-Free SVS')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--phone_path', type=str, default=None, help='Path to phoneme sequence file')
    parser.add_argument('--f0_path', type=str, default=None, help='Path to F0 contour file')
    parser.add_argument('--duration_path', type=str, default=None, help='Path to duration file')
    parser.add_argument('--midi_path', type=str, default=None, help='Path to MIDI sequence file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs (overrides config)')
    parser.add_argument('--output_name', type=str, default='output', help='Base name for output files')
    parser.add_argument('--vocoder_path', type=str, default=None, help='Path to vocoder model (overrides config)')
    parser.add_argument('--generate_audio', action='store_true', help='Generate audio using vocoder')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on')
    parser.add_argument('--adjust_length', action='store_true', 
                        help='Adjust output mel length to match input length divided by total upsampling factor')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    output_dir = args.output_dir or config.get('inference', {}).get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config)
    model = model.to(device)
    
    # Prepare input data
    print("Preparing input data...")
    phone_ids, f0, durations, midi_ids = prepare_input_data(
        args.phone_path, args.f0_path, args.duration_path, args.midi_path, device
    )
    
    # Calculate expected output length based on input length and upsampling factor
    input_length = phone_ids.shape[1]
    
    # Calculate total upsampling factor from configuration
    total_upsampling = 1
    for factor in config['model']['upsampling_factors']:
        total_upsampling *= factor
        
    expected_length = input_length * total_upsampling
    print(f"Input sequence length: {input_length}")
    print(f"Total upsampling factor: {total_upsampling}x")
    print(f"Expected output length: {expected_length}")
    
    # Generate mel spectrogram
    print("Generating mel spectrogram...")
    with torch.no_grad():
        mel_output = model(phone_ids, f0, durations, midi_ids)
        
        # Adjust length if requested
        if args.adjust_length:
            print(f"Adjusting mel length from {mel_output.shape[-1]} to {input_length}...")
            mel_output = model.adjust_mel_length(mel_output)
            print(f"Adjusted mel shape: {mel_output.shape}")
    
    # Save mel spectrogram
    mel_output_path = os.path.join(output_dir, f"{args.output_name}.npy")
    save_mel_spectrogram(mel_output, mel_output_path)
    
    # Generate audio if requested
    if args.generate_audio:
        print("Generating audio with vocoder...")
        
        # Load vocoder
        vocoder_path = args.vocoder_path or config.get('inference', {}).get('vocoder_path')
        if not vocoder_path:
            print("Error: Vocoder path not specified!")
            return
        
        vocoder = load_vocoder(vocoder_path, device)
        
        # Generate audio
        audio = vocoder.generate(mel_output)
        
        # Save audio
        audio = audio.squeeze().cpu().numpy()
        audio_output_path = os.path.join(output_dir, f"{args.output_name}.wav")
        sf.write(audio_output_path, audio, config['audio']['sample_rate'])
        
        print(f"Saved audio to {audio_output_path}")
    
    print("Inference completed!")

if __name__ == "__main__":
    main()