# inference.py

import torch
import torch.nn.functional as F
import io
import os
import yaml
import argparse
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, TYPE_CHECKING
import pytorch_lightning as pl # For seed_everything

# Import local modules
from utils.plotting import plot_spectrograms_to_figure
from data.dataset import H5Dataset, H5FileManager # Import dataset components
from models.progressive_svs import ProgressiveSVS # Import the model directly

# Remove TYPE_CHECKING block as we import ProgressiveSVS directly now
# if TYPE_CHECKING:
#     from models.progressive_svs import ProgressiveSVS

def inference(
    model: 'ProgressiveSVS',
    f0: torch.Tensor,
    phone_label: torch.Tensor,
    phone_duration: torch.Tensor,
    midi_label: torch.Tensor,
    unvoiced_flag: torch.Tensor,
    mel_spec_gt: torch.Tensor, # Ground truth for comparison
    length: torch.Tensor,      # Original length before padding (scalar tensor)
    config: dict,              # Model configuration (needed for potential stage info, though we assume stage 3)
    device: torch.device
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Performs inference using the ProgressiveSVS model, generates a comparison
    plot of the predicted vs. ground truth mel spectrogram (without padding),
    and returns the plot and the unpadded predicted spectrogram.

    Assumes the model is configured to output at Stage 3 (full frequency resolution).
    Assumes input tensors are for a single batch item (B=1).

    Args:
        model: Trained ProgressiveSVS model instance.
        f0: F0 contour tensor (B=1, T_orig, 1).
        phone_label: Phone label tensor (B=1, T_orig).
        phone_duration: Phone duration tensor (B=1, T_orig).
        midi_label: MIDI label tensor (B=1, T_orig, 1 or B=1, T_orig).
        unvoiced_flag: Unvoiced flag tensor (B=1, T_orig, 1 or B=1, T_orig).
        mel_spec_gt: Ground truth mel spectrogram (B=1, T_orig, F_full).
        length: Original, unpadded length of the sequence (scalar tensor).
        config: Model configuration dictionary.
        device: The torch device ('cpu' or 'cuda:X') to run inference on.

    Returns:
        A tuple containing:
        - PIL.Image.Image: Comparison plot (predicted vs GT, unpadded).
        - torch.Tensor: Predicted mel spectrogram, unpadded (T_unpadded, F_full).
    """
    model.eval()
    model.to(device)

    # Move inputs to the specified device
    f0 = f0.to(device)
    phone_label = phone_label.to(device)
    phone_duration = phone_duration.to(device)
    midi_label = midi_label.to(device)
    unvoiced_flag = unvoiced_flag.to(device)
    mel_spec_gt = mel_spec_gt.to(device)
    length = length.to(device) # Keep length on device for calculations if needed

    with torch.no_grad():
        # --- Forward Pass ---
        # Assumes model's current_stage is set correctly (e.g., 3 for full res)
        # Output shape: (B=1, T_downsampled, F_out)
        mel_pred = model(f0, phone_label, phone_duration, midi_label, unvoiced_flag)

        # --- Generate Target Mel (Matching Prediction Resolution) ---
        # We need the target mel at the same temporal resolution as the prediction
        original_max_len = mel_spec_gt.shape[1] # T_orig
        target_time_dim = mel_pred.shape[1]     # T_downsampled
        freq_dim = mel_pred.shape[2]            # F_out (should be F_full for stage 3)

        # Ensure mel_spec_gt has 3 dims (B, T, F)
        if mel_spec_gt.dim() != 3:
             raise ValueError(f"Expected mel_spec_gt to have 3 dimensions (B, T, F), but got shape {mel_spec_gt.shape}")
        if mel_spec_gt.shape[0] != 1:
             print(f"Warning: Expected batch size 1 for mel_spec_gt, but got {mel_spec_gt.shape[0]}. Using first item.")
             mel_spec_gt = mel_spec_gt[0:1] # Keep batch dim

        # Reshape GT for interpolation: (B=1, F_full, T_orig) -> (B=1, 1, F_full, T_orig)
        mel_spec_gt_permuted = mel_spec_gt.permute(0, 2, 1)
        mel_spec_gt_4d = mel_spec_gt_permuted.unsqueeze(1)

        # Interpolate GT to match prediction's temporal resolution (T_downsampled)
        # We assume the frequency dimension (F_out) already matches F_full (Stage 3)
        mel_target = F.interpolate(
            mel_spec_gt_4d,
            size=(freq_dim, target_time_dim), # Target shape (F_full, T_downsampled)
            mode='bilinear',
            align_corners=False
        ).squeeze(1) # Back to (B=1, F_full, T_downsampled)

        # Permute back to (B=1, T_downsampled, F_full)
        mel_target = mel_target.permute(0, 2, 1)

        # --- Calculate Unpadded Length in Target Time Dimension ---
        # Ensure length is a scalar
        if not (length.ndim == 0 or (length.ndim == 1 and length.numel() == 1)):
             raise ValueError(f"Expected length to be a scalar tensor, but got shape {length.shape}")
        original_length_scalar = length.item() # Length derived from duration sum
        target_length = int(original_length_scalar // model.downsample_stride)

        # --- Slice to Remove Padding ---
        # Remove batch dimension (B=1) and slice time dimension
        #mel_pred = mel_pred.T
        #mel_target = mel_target.T
        mel_pred_unpadded = mel_pred[0, :target_length, :]   # Shape: (T_unpadded, F_full)
        mel_target_unpadded = mel_target[0, :target_length, :] # Shape: (T_unpadded, F_full)

        # --- Generate Comparison Plot ---
        # The plotting function handles tensor conversion internally.
        # Pass the tensors directly: ground_truth (target) first, then prediction.
        # Pass the tensors directly: ground_truth (target) first, then prediction.
        vmin = mel_pred_unpadded.min()
        vmax = mel_pred_unpadded.max()

        mel_pred_unpadded = mel_pred_unpadded.T
        mel_target_unpadded = mel_target_unpadded.T

        fig = plot_spectrograms_to_figure(mel_target_unpadded, mel_pred_unpadded, vmin, vmax)

        # --- Convert Figure to PIL Image ---
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        buf.seek(0)
        image = Image.open(buf)

        # Return the image and the unpadded prediction tensor (still on original device)
        # Return the image and the unpadded prediction tensor (still on original device)
        return image, mel_pred_unpadded


if __name__ == "__main__":
    pl.seed_everything(42, workers=True) # Set seed for reproducibility

    parser = argparse.ArgumentParser(description="Run inference on a single sample from the dataset.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file (e.g., config/model.yaml).')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.ckpt) file.')
    parser.add_argument('--sample_index', type=int, required=True, help='Index of the sample to load from the dataset.')
    parser.add_argument('--output_image', type=str, required=True, help='Path to save the output comparison image (e.g., output/comparison.png).')
    parser.add_argument('--output_tensor', type=str, default=None, help='Optional path to save the predicted mel tensor (.pt).')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Device to use (cpu or cuda). Auto-detects if not specified.')
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        exit(1)

    # --- 2. Determine Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Prepare Config for Inference ---
    # Set stage to 3 for full resolution output during inference
    config['model']['current_stage'] = 3
    print(f"Set model stage for inference to: {config['model']['current_stage']}")

    # Load vocab_size from HDF5 file attribute
    h5_path = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
    h5_manager = H5FileManager.get_instance()
    try:
        h5_file = h5_manager.get_file(h5_path)
        if 'vocab_size' in h5_file.attrs:
            config['model']['vocab_size'] = int(h5_file.attrs['vocab_size'])
            print(f"Read vocab_size from HDF5 attribute: {config['model']['vocab_size']}")
        else:
            raise ValueError(f"'vocab_size' attribute not found in HDF5 file: {h5_path}")
    except Exception as e:
        print(f"Error reading vocab_size from HDF5 file '{h5_path}': {e}")
        h5_manager.close_all()
        exit(1)
    # Keep HDF5 file open via manager for dataset loading

    # --- 4. Load Model ---
    print(f"Loading model from checkpoint: {args.checkpoint}")
    try:
        # Load model, ensuring config overrides are applied
        model = ProgressiveSVS.load_from_checkpoint(
            args.checkpoint,
            config=config, # Pass the modified config
            map_location=device # Load directly to the target device
        )
        model.eval() # Set to evaluation mode
        model.to(device) # Ensure model is on the correct device
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        h5_manager.close_all()
        exit(1)
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        h5_manager.close_all()
        exit(1)

    # --- 5. Load Data Sample ---
    print(f"Loading sample index {args.sample_index} from dataset: {h5_path}")
    try:
        # Define data keys needed based on config (mirroring DataModule)
        data_keys = {
            'mel_spectrogram': config['data']['mel_key'],
            'f0_contour': config['data']['f0_key'],
            'phone_sequence': config['data']['phoneme_key'], # Frame-level IDs from HDF5
            'adjusted_durations': config['data']['duration_key'], # Phoneme-level durations
            'phone_sequence_ids': 'phone_sequence_ids', # Phoneme-level IDs
            'midi_pitch_estimated': 'midi_pitch_estimated',
            'original_unpadded_length': 'original_unpadded_length', # Add key for true length
            'voiced_mask': 'voiced_mask',
            'unvoiced_flag': 'unvoiced_flag'
        }
        dataset = H5Dataset(h5_path=h5_path, data_keys=data_keys, lazy_load=True) # Use lazy load

        if args.sample_index < 0 or args.sample_index >= len(dataset):
             raise IndexError(f"Sample index {args.sample_index} is out of bounds for dataset size {len(dataset)}.")

        sample_dict = dataset[args.sample_index]
        print(f"DEBUG: Keys in loaded sample_dict: {list(sample_dict.keys())}") # Added for debugging
        if sample_dict is None:
            raise ValueError(f"Failed to load sample at index {args.sample_index}.")

        print(f"Sample {args.sample_index} loaded successfully.")

    except FileNotFoundError:
         print(f"Error: HDF5 dataset file not found at '{h5_path}'.")
         h5_manager.close_all()
         exit(1)
    except Exception as e:
        print(f"Error loading data sample: {e}")
        h5_manager.close_all()
        exit(1)

    # --- 6. Prepare Inputs for Inference Function ---
    try:
        # Extract tensors using config keys and rename/prepare
        mel_spec_gt_orig = sample_dict[config['data']['mel_key']]
        f0_orig = sample_dict[config['data']['f0_key']]
        midi_label_orig = sample_dict['midi_pitch_estimated']
        unvoiced_flag_orig = sample_dict['unvoiced_flag']
        # Note: 'phoneme' key from HDF5 contains frame-level IDs, not needed directly for model input
        # We need phoneme-level IDs and durations to create the frame-level 'phone_label'
        phoneme_level_ids = sample_dict['phone_sequence_ids']
        phone_duration_orig = sample_dict[config['data']['duration_key']] # Phoneme-level durations

        # Load the true original unpadded length stored during preprocessing
        if 'original_unpadded_length' not in sample_dict:
             raise KeyError("Dataset does not contain 'original_unpadded_length'. Please re-run preprocessing.")
        # Ensure it's a scalar tensor
        original_length = torch.tensor(sample_dict['original_unpadded_length'], dtype=torch.long)
        if original_length.ndim > 0: # If it was saved as an array, get the scalar value
             original_length = original_length.item()
             original_length = torch.tensor(original_length, dtype=torch.long) # Convert back to scalar tensor

        # Expand phoneme IDs to frame-level 'phone_label'
        target_frame_len = mel_spec_gt_orig.shape[0] # Use PADDED length for model input consistency
        phone_duration_orig = torch.clamp(phone_duration_orig, min=0)
        total_duration = phone_duration_orig.sum()

        if total_duration > 0:
            # Ensure phoneme_level_ids and phone_duration_orig are 1D tensors
            if phoneme_level_ids.dim() > 1: phoneme_level_ids = phoneme_level_ids.squeeze()
            if phone_duration_orig.dim() > 1: phone_duration_orig = phone_duration_orig.squeeze()

            # Check if lengths match before repeat_interleave
            if len(phoneme_level_ids) != len(phone_duration_orig):
                 raise ValueError(f"Mismatch between phoneme ID count ({len(phoneme_level_ids)}) and duration count ({len(phone_duration_orig)})")

            # Cast durations to long type for repeat_interleave
            expanded_phonemes = torch.repeat_interleave(phoneme_level_ids, phone_duration_orig.long())
            current_expanded_len = expanded_phonemes.shape[0]

            # Pad or truncate expanded sequence if needed (should ideally match original_length)
            if current_expanded_len < target_frame_len:
                padding_size = target_frame_len - current_expanded_len
                # print(f"Warning: Expanded phoneme length ({current_expanded_len}) is less than target frame length ({target_frame_len}). Padding with 0.")
                expanded_phonemes = torch.nn.functional.pad(expanded_phonemes, (0, padding_size), mode='constant', value=0)
            elif current_expanded_len > target_frame_len:
                # print(f"Warning: Expanded phoneme length ({current_expanded_len}) exceeds target frame length ({target_frame_len}). Truncating.")
                expanded_phonemes = expanded_phonemes[:target_frame_len]
        else:
            expanded_phonemes = torch.zeros(target_frame_len, dtype=torch.long)

        phone_label_orig = expanded_phonemes # This is the frame-level label

        # Add batch dimension (unsqueeze(0))
        f0_batch = f0_orig.unsqueeze(0)
        phone_label_batch = phone_label_orig.unsqueeze(0)
        phone_duration_batch = phone_duration_orig.unsqueeze(0) # Keep phoneme-level duration for model input
        midi_label_batch = midi_label_orig.unsqueeze(0)
        unvoiced_flag_batch = unvoiced_flag_orig.unsqueeze(0)
        mel_spec_gt_batch = mel_spec_gt_orig.unsqueeze(0)
        # length remains a scalar tensor

        print("Input tensors prepared for inference function.")

    except KeyError as e:
        print(f"Error: Missing expected key '{e}' in the loaded sample dictionary or config.")
        h5_manager.close_all()
        exit(1)
    except Exception as e:
        print(f"Error preparing input tensors: {e}")
        import traceback
        traceback.print_exc()
        h5_manager.close_all()
        exit(1)


    # --- 7. Run Inference ---
    print("Running inference...")
    try:
        comparison_image, predicted_mel_unpadded = inference(
            model=model,
            f0=f0_batch,
            phone_label=phone_label_batch,
            phone_duration=phone_duration_batch, # Pass phoneme-level duration
            midi_label=midi_label_batch,
            unvoiced_flag=unvoiced_flag_batch,
            mel_spec_gt=mel_spec_gt_batch,
            length=original_length, # Pass scalar original length
            config=config,
            device=device
        )
        print("Inference completed.")
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()
        h5_manager.close_all()
        exit(1)

    # --- 8. Save Outputs ---
    # Ensure output directory exists for image
    output_image_dir = os.path.dirname(args.output_image)
    if output_image_dir: # Create dir only if path includes one
        os.makedirs(output_image_dir, exist_ok=True)

    # Save image
    try:
        comparison_image.save(args.output_image)
        print(f"Comparison image saved to: {args.output_image}")
    except Exception as e:
        print(f"Error saving comparison image: {e}")

    # Save tensor if path provided
    if args.output_tensor:
        output_tensor_dir = os.path.dirname(args.output_tensor)
        if output_tensor_dir: # Create dir only if path includes one
            os.makedirs(output_tensor_dir, exist_ok=True)
        try:
            # Move tensor to CPU before saving if it's not already
            torch.save(predicted_mel_unpadded.cpu(), args.output_tensor)
            print(f"Predicted mel tensor saved to: {args.output_tensor}")
        except Exception as e:
            print(f"Error saving predicted tensor: {e}")

    # --- 9. Cleanup ---
    h5_manager.close_all()
    print("HDF5 file manager closed.")
    print("Script finished.")