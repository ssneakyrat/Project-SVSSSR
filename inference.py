import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sys

sys.path.append('.')

from utils.utils import load_config, extract_f0, normalize_mel_spectrogram, denormalize_mel_spectrogram
from models.progressive_svs import ProgressiveSVS

def load_model(checkpoint_path, config_path="config/model.yaml"):
    """Load model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found")
        return None
    
    config = load_config(config_path)
    
    try:
        model = ProgressiveSVS(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Successfully loaded model from {checkpoint_path}")
            
            # Get the stage from the model
            if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
                model_config = checkpoint['hyper_parameters']['config']
                model.current_stage = model_config['model']['current_stage']
            
            return model
        else:
            print(f"Error: No state_dict found in checkpoint {checkpoint_path}")
            return None
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_example_inputs(config, time_frames=864):
    """Create a test input for inference"""
    # Create dummy f0 contour (sine wave)
    f0_min = config['audio']['f0_min']
    f0_max = config['audio']['f0_max']
    t = np.linspace(0, 10, time_frames)
    f0 = f0_min + (f0_max - f0_min) * 0.5 * (1 + np.sin(2 * np.pi * 0.2 * t))
    
    # Create dummy phone labels (alternating phonemes)
    phone_label = np.zeros(time_frames, dtype=np.int64)
    for i in range(8):  # 8 segments
        start = i * (time_frames // 8)
        end = (i + 1) * (time_frames // 8)
        phone_label[start:end] = i % 20  # Using 20 phonemes
    
    # Create dummy durations
    phone_durations = np.ones(20, dtype=np.float32) * (time_frames // 20)
    
    # Create dummy MIDI labels (scale pattern)
    midi_label = np.zeros(time_frames, dtype=np.int64)
    for i in range(8):  # 8 segments
        start = i * (time_frames // 8)
        end = (i + 1) * (time_frames // 8)
        midi_label[start:end] = 60 + (i % 12)  # C4 to B4
    
    # Convert to tensors
    f0 = torch.from_numpy(f0).float().unsqueeze(0)  # [1, T]
    phone_label = torch.from_numpy(phone_label).long().unsqueeze(0)  # [1, T]
    phone_durations = torch.from_numpy(phone_durations).float().unsqueeze(0)  # [1, D]
    midi_label = torch.from_numpy(midi_label).long().unsqueeze(0)  # [1, T]
    
    return f0, phone_label, phone_durations, midi_label

def visualize_mel(mel_spectrogram, title="Generated Mel Spectrogram", save_path=None):
    """Visualize and optionally save a mel spectrogram"""
    plt.figure(figsize=(12, 6))
    
    # If 4D tensor [B, C, H, W], convert to 2D
    if mel_spectrogram.dim() == 4:
        mel_spectrogram = mel_spectrogram.squeeze(0).squeeze(0)
    
    # If 3D tensor [B, H, W], convert to 2D
    if mel_spectrogram.dim() == 3:
        mel_spectrogram = mel_spectrogram.squeeze(0)
    
    plt.imshow(mel_spectrogram.numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Bin')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inference with SVS Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output/generated_mel.png', help='Path to save output visualization')
    parser.add_argument('--wav', type=str, default=None, help='Optional path to WAV file for F0 extraction')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.config)
    if model is None:
        return
    
    # Switch to eval mode
    model.eval()
    
    # Load config
    config = load_config(args.config)
    
    # Create example inputs or load from WAV file
    if args.wav and os.path.exists(args.wav):
        print(f"Using F0 from WAV file: {args.wav}")
        # Extract F0 from WAV file
        f0 = extract_f0(args.wav, config)
        
        # Ensure F0 length matches model requirements
        time_frames = config['model']['time_frames']
        if len(f0) > time_frames:
            f0 = f0[:time_frames]
        elif len(f0) < time_frames:
            pad_length = time_frames - len(f0)
            f0 = np.pad(f0, (0, pad_length), mode='constant')
        
        # Create other inputs
        f0 = torch.from_numpy(f0).float().unsqueeze(0)  # [1, T]
        _, phone_label, phone_durations, midi_label = prepare_example_inputs(config, time_frames)
    else:
        print("Using synthetic example inputs")
        f0, phone_label, phone_durations, midi_label = prepare_example_inputs(config)
    
    # Run inference
    print(f"Running inference with model at stage {model.current_stage}...")
    with torch.no_grad():
        # Calculate sequence length based on F0
        # Create a simple binary "has content" mask from F0
        f0_mask = (f0[0] > 0.0).float()
        # Find the last non-zero position
        if torch.any(f0_mask):
            seq_length = torch.argwhere(f0_mask)[-1].item() + 1
        else:
            seq_length = f0.shape[1]  # Use full length if no content detected
        
        # Create a tensor for lengths
        lengths = torch.tensor([seq_length], device=f0.device)
        
        # Pass lengths to model
        mel_output = model(f0, phone_label, phone_durations, midi_label, lengths)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Get dimension text based on stage
    if model.current_stage == 1:
        dim_text = "20×216"
    elif model.current_stage == 2:
        dim_text = "40×432"
    else:
        dim_text = "80×864"
    
    # Visualize the output
    print("Visualizing output...")
    visualize_mel(
        mel_output, 
        title=f"Generated Mel Spectrogram (Stage {model.current_stage}: {dim_text})",
        save_path=args.output
    )
    
    # Print expected parameter counts
    params = sum(p.numel() for p in model.parameters())
    low_res_params = sum(p.numel() for p in model.low_res_model.parameters())
    mid_res_params = sum(p.numel() for p in model.mid_res_upsampler.parameters())
    high_res_params = sum(p.numel() for p in model.high_res_upsampler.parameters())
    
    print(f"Model statistics:")
    print(f"  Total parameters: {params:,}")
    print(f"  Low-res model parameters: {low_res_params:,} (~160K expected)")
    print(f"  Mid-res upsampler parameters: {mid_res_params:,} (~42K expected)")
    print(f"  High-res upsampler parameters: {high_res_params:,} (~28K expected)")
    
    # Print memory usage estimate
    vram_estimate = params * 4 / 1024 / 1024  # 4 bytes per parameter, convert to MB
    print(f"  Estimated VRAM for parameters: {vram_estimate:.2f} MB")
    print(f"  Expected VRAM usage with batch size 16: ~2-3 GB")

if __name__ == "__main__":
    main()