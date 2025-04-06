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
from typing import Tuple, TYPE_CHECKING, Optional # Added Optional
import pytorch_lightning as pl # For seed_everything
import logging # Import logging
import traceback # Import traceback

# Import local modules
from utils.plotting import plot_spectrograms_to_figure, plot_single_spectrogram_to_figure # Added plot_single_spectrogram_to_figure
from utils.utils import setup_logging # Import logging setup
from data.dataset import H5Dataset, H5FileManager # Import dataset components
from models.progressive_svs import ProgressiveSVS # Import the model directly

logger = logging.getLogger(__name__) # Get module-level logger

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
    length: torch.Tensor,                       # Original length before padding (scalar tensor)
    config: dict,                               # Model configuration (needed for potential stage info, though we assume stage 3)
    device: torch.device,
    mel_spec_gt: Optional[torch.Tensor] = None  # Ground truth (optional, moved to end)
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Performs inference using the ProgressiveSVS model.
    If ground truth (mel_spec_gt) is provided, generates a comparison plot.
    Otherwise, generates a plot of only the predicted spectrogram.
    Returns the plot image and the unpadded predicted spectrogram.

    Assumes the model is configured to output at Stage 3 (full frequency resolution).
    Assumes input tensors are for a single batch item (B=1).

    Args:
        model: Trained ProgressiveSVS model instance.
        f0: F0 contour tensor (B=1, T_orig, 1).
        phone_label: Phone label tensor (B=1, T_orig).
        phone_duration: Phone duration tensor (B=1, T_orig).
        midi_label: MIDI label tensor (B=1, T_orig, 1 or B=1, T_orig).
        unvoiced_flag: Unvoiced flag tensor (B=1, T_orig, 1 or B=1, T_orig).
        length: Original, unpadded length of the sequence (scalar tensor).
        config: Model configuration dictionary.
        device: The torch device ('cpu' or 'cuda:X') to run inference on.
        mel_spec_gt: Optional ground truth mel spectrogram (B=1, T_orig, F_full). If None, comparison is skipped.

    Returns:
        A tuple containing:
        - PIL.Image.Image: Plot image (comparison or prediction-only, unpadded).
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
    if mel_spec_gt is not None:
        mel_spec_gt = mel_spec_gt.to(device)
    length = length.to(device) # Keep length on device for calculations if needed

    with torch.no_grad():
        # --- Forward Pass ---
        # Assumes model's current_stage is set correctly (e.g., 3 for full res)
        # Output shape: (B=1, T_downsampled, F_out)
        mel_pred = model(f0, phone_label, phone_duration, midi_label, unvoiced_flag)

        # --- Calculate Unpadded Length in Target Time Dimension ---
        # Ensure length is a scalar
        if not (length.ndim == 0 or (length.ndim == 1 and length.numel() == 1)):
             raise ValueError(f"Expected length to be a scalar tensor, but got shape {length.shape}")
        original_length_scalar = length.item() # Length derived from duration sum
        target_length = int(original_length_scalar // model.downsample_stride) # Unpadded length in downsampled time dim

        # --- Slice Prediction to Remove Padding ---
        # Remove batch dimension (B=1) and slice time dimension
        mel_pred_unpadded = mel_pred[0, :target_length, :]   # Shape: (T_unpadded, F_full)

        # Determine min/max for plotting color scale based on prediction
        vmin = mel_pred_unpadded.min().item() # Use .item() to get scalar value
        vmax = mel_pred_unpadded.max().item()

        # Transpose prediction for plotting (Freq, Time)
        mel_pred_unpadded = mel_pred_unpadded.T

        # --- Process Ground Truth (if available) and Generate Plot ---
        if mel_spec_gt is not None:
            # --- Generate Target Mel (Matching Prediction Resolution) ---
            target_time_dim = mel_pred.shape[1]     # T_downsampled (from model output)
            freq_dim = mel_pred.shape[2]            # F_out (should be F_full for stage 3)

            # Ensure mel_spec_gt has 3 dims (B, T, F)
            if mel_spec_gt.dim() != 3:
                 raise ValueError(f"Expected mel_spec_gt to have 3 dimensions (B, T, F), but got shape {mel_spec_gt.shape}")
            if mel_spec_gt.shape[0] != 1:
                 logger.warning(f"Expected batch size 1 for mel_spec_gt, but got {mel_spec_gt.shape[0]}. Using first item.")
                 mel_spec_gt = mel_spec_gt[0:1] # Keep batch dim

            # Reshape GT for interpolation: (B=1, F_full, T_orig) -> (B=1, 1, F_full, T_orig)
            mel_spec_gt_permuted = mel_spec_gt.permute(0, 2, 1)
            mel_spec_gt_4d = mel_spec_gt_permuted.unsqueeze(1)

            # Interpolate GT to match prediction's temporal resolution (T_downsampled)
            mel_target = F.interpolate(
                mel_spec_gt_4d,
                size=(freq_dim, target_time_dim), # Target shape (F_full, T_downsampled)
                mode='bilinear',
                align_corners=False
            ).squeeze(1) # Back to (B=1, F_full, T_downsampled)

            # Permute back to (B=1, T_downsampled, F_full)
            mel_target = mel_target.permute(0, 2, 1)

            # --- Slice Target to Remove Padding ---
            mel_target_unpadded = mel_target[0, :target_length, :] # Shape: (T_unpadded, F_full)

            # Transpose target for plotting (Freq, Time)
            mel_target_unpadded = mel_target_unpadded.T

            # --- Generate Comparison Plot ---
            fig = plot_spectrograms_to_figure(
                ground_truth=mel_target_unpadded,
                prediction=mel_pred_unpadded,
                title="Spectrogram Comparison (Unpadded)",
                vmin=vmin, # Use calculated vmin/vmax
                vmax=vmax
            )
        else:
            # --- Generate Prediction-Only Plot ---
            fig = plot_single_spectrogram_to_figure(
                prediction=mel_pred_unpadded,
                title="Predicted Spectrogram (Unpadded)",
                vmin=vmin, # Use calculated vmin/vmax
                vmax=vmax
            )

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
    # Setup logging first
    setup_logging(level=logging.INFO)
    logger.info("Starting inference script...")

    pl.seed_everything(42, workers=True) # Set seed for reproducibility
    logger.info("Set random seed to 42.")

    parser = argparse.ArgumentParser(description="Run inference on a single sample from the dataset.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file (e.g., config/model.yaml).')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.ckpt) file.')
    parser.add_argument('--sample_index', type=int, required=True, help='Index of the sample to load from the dataset.')
    parser.add_argument('--output_image', type=str, required=True, help='Path to save the output comparison image (e.g., output/comparison.png).')
    parser.add_argument('--output_tensor', type=str, default=None, help='Optional path to save the predicted mel tensor (.pt).')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Device to use (cpu or cuda). Auto-detects if not specified.')
    parser.add_argument('--use_ground_truth', action='store_true', help='Load ground truth mel spectrogram for comparison plotting.')
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        exit(1)

    # --- 2. Determine Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 3. Prepare Config for Inference ---
    # Set stage to 3 for full resolution output during inference
    config['model']['current_stage'] = 3
    logger.info(f"Set model stage for inference to: {config['model']['current_stage']}")

    # Load vocab_size from HDF5 file attribute
    h5_path = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
    h5_manager = H5FileManager.get_instance()
    try:
        h5_file = h5_manager.get_file(h5_path)
        if 'vocab_size' in h5_file.attrs:
            config['model']['vocab_size'] = int(h5_file.attrs['vocab_size'])
            logger.info(f"Read vocab_size from HDF5 attribute: {config['model']['vocab_size']}")
        else:
            raise ValueError(f"'vocab_size' attribute not found in HDF5 file: {h5_path}")
    except Exception as e:
        logger.error(f"Error reading vocab_size from HDF5 file '{h5_path}': {e}")
        h5_manager.close_all()
        exit(1)
    # Keep HDF5 file open via manager for dataset loading

    # --- 4. Load Model ---
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    try:
        # Load model, ensuring config overrides are applied
        model = ProgressiveSVS.load_from_checkpoint(
            args.checkpoint,
            config=config, # Pass the modified config
            map_location=device # Load directly to the target device
        )
        model.eval() # Set to evaluation mode
        model.to(device) # Ensure model is on the correct device
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {args.checkpoint}")
        h5_manager.close_all()
        exit(1)
    except Exception as e:
        logger.error(f"Error loading model from checkpoint: {e}")
        h5_manager.close_all()
        exit(1)

    # --- 5. Load Data Sample ---
    logger.info(f"Loading sample index {args.sample_index} from dataset: {h5_path}")
    try:
        # Define data keys needed based on config (mirroring DataModule)
        # Define base data keys needed regardless of ground truth
        data_keys = {
            'f0_contour': config['data']['f0_key'],
            # 'phone_sequence': config['data']['phoneme_key'], # Frame-level IDs not directly used
            'adjusted_durations': config['data']['duration_key'], # Phoneme-level durations
            'phone_sequence_ids': 'phone_sequence_ids', # Phoneme-level IDs
            'midi_pitch_estimated': 'midi_pitch_estimated',
            'original_unpadded_length': 'original_unpadded_length', # Add key for true length
            # 'voiced_mask': 'voiced_mask', # Not directly used by model? Check ProgressiveSVS input
            'unvoiced_flag': 'unvoiced_flag'
        }
        # Conditionally add mel spectrogram key if ground truth is needed
        # Conditionally add mel spectrogram key ONLY if ground truth comparison is requested
        if args.use_ground_truth:
            logger.info("Ground truth comparison requested, adding mel spectrogram key.")
            data_keys['mel_spectrogram'] = config['data']['mel_key']
        else:
            logger.info("Running inference without ground truth (default).")
        dataset = H5Dataset(h5_path=h5_path, data_keys=data_keys, lazy_load=True) # Use lazy load

        if args.sample_index < 0 or args.sample_index >= len(dataset):
             raise IndexError(f"Sample index {args.sample_index} is out of bounds for dataset size {len(dataset)}.")

        sample_dict = dataset[args.sample_index]
        logger.debug(f"Keys in loaded sample_dict: {list(sample_dict.keys())}")
        if sample_dict is None:
            raise ValueError(f"Failed to load sample at index {args.sample_index}.")

        logger.info(f"Sample {args.sample_index} loaded successfully.")

    except FileNotFoundError:
         logger.error(f"HDF5 dataset file not found at '{h5_path}'.")
         h5_manager.close_all()
         exit(1)
    except Exception as e:
        logger.error(f"Error loading data sample: {e}")
        h5_manager.close_all()
        exit(1)

    # --- 6. Prepare Inputs for Inference Function ---
    try:
        # Extract tensors common to both modes
        f0_orig = sample_dict[config['data']['f0_key']]
        if f0_orig is None: raise KeyError(f"Required key '{config['data']['f0_key']}' (f0_contour) not found in sample.")
        midi_label_orig = sample_dict['midi_pitch_estimated']
        if midi_label_orig is None: raise KeyError("'midi_pitch_estimated' not found in sample.")
        unvoiced_flag_orig = sample_dict['unvoiced_flag']
        if unvoiced_flag_orig is None: raise KeyError("'unvoiced_flag' not found in sample.")
        phoneme_level_ids = sample_dict['phone_sequence_ids']
        if phoneme_level_ids is None: raise KeyError("'phone_sequence_ids' not found in sample.")
        phone_duration_orig = sample_dict[config['data']['duration_key']] # Phoneme-level durations
        if phone_duration_orig is None: raise KeyError(f"Required key '{config['data']['duration_key']}' (duration) not found in sample.")

        # Initialize GT variables
        mel_spec_gt_orig = None
        mel_spec_gt_batch = None

        # Load GT only if requested
        # Load GT only if requested via flag
        if args.use_ground_truth:
            mel_key = config['data']['mel_key']
            if mel_key not in sample_dict or sample_dict[mel_key] is None:
                raise KeyError(f"Ground truth comparison requested via flag, but key '{mel_key}' not found or is None in sample.")
            logger.info("Loading ground truth mel spectrogram for comparison.")
            mel_spec_gt_orig = sample_dict[mel_key]
            mel_spec_gt_batch = mel_spec_gt_orig.unsqueeze(0) # Add batch dim for GT

        # Load the true original unpadded length stored during preprocessing
        if 'original_unpadded_length' not in sample_dict:
             raise KeyError("Dataset does not contain 'original_unpadded_length'. Please re-run preprocessing.")
        # Ensure it's a scalar tensor
        original_length = torch.tensor(sample_dict['original_unpadded_length'], dtype=torch.long)
        if original_length.ndim > 0: # If it was saved as an array, get the scalar value
             original_length = original_length.item()
             original_length = torch.tensor(original_length, dtype=torch.long) # Convert back to scalar tensor

        # Determine target frame length (padded length) for expanding phonemes
        # Use GT shape if available, otherwise use F0 shape (assuming they are padded identically)
        if mel_spec_gt_orig is not None:
            target_frame_len = mel_spec_gt_orig.shape[0]
            logger.debug(f"Using target_frame_len from mel_spec_gt: {target_frame_len}")
        else:
            target_frame_len = f0_orig.shape[0] # Assumes f0 is padded to the same length
            logger.debug(f"Using target_frame_len from f0_contour: {target_frame_len}")

        # Expand phoneme IDs to frame-level 'phone_label' using target_frame_len
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
        # mel_spec_gt_batch is already prepared (or None)
        # length remains a scalar tensor

        logger.info("Input tensors prepared for inference function.")

    except KeyError as e:
        logger.error(f"Missing expected key '{e}' in the loaded sample dictionary or config.")
        h5_manager.close_all()
        exit(1)
    except Exception as e:
        logger.error(f"Error preparing input tensors: {e}")
        logger.error(traceback.format_exc()) # Log traceback
        h5_manager.close_all()
        exit(1)


    # --- 7. Run Inference ---
    logger.info("Running inference...")
    try:
        # Call inference, passing mel_spec_gt_batch which might be None
        comparison_image, predicted_mel_unpadded = inference(
            model=model,
            f0=f0_batch,
            phone_label=phone_label_batch,
            phone_duration=phone_duration_batch, # Pass phoneme-level duration
            midi_label=midi_label_batch,
            unvoiced_flag=unvoiced_flag_batch,
            length=original_length, # Pass scalar original length
            config=config,
            device=device,
            mel_spec_gt=mel_spec_gt_batch # Pass GT (or None) as the last argument
        )
        logger.info("Inference completed.")
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        logger.error(traceback.format_exc()) # Log traceback
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
        logger.info(f"Output image saved to: {args.output_image}")
    except Exception as e:
        logger.error(f"Error saving comparison image: {e}")

    # Save tensor if path provided
    if args.output_tensor:
        output_tensor_dir = os.path.dirname(args.output_tensor)
        if output_tensor_dir: # Create dir only if path includes one
            os.makedirs(output_tensor_dir, exist_ok=True)
        try:
            # Move tensor to CPU before saving if it's not already
            torch.save(predicted_mel_unpadded.cpu(), args.output_tensor)
            logger.info(f"Predicted mel tensor saved to: {args.output_tensor}")
        except Exception as e:
            logger.error(f"Error saving predicted tensor: {e}")

    # --- 9. Cleanup ---
    h5_manager.close_all()
    logger.info("HDF5 file manager closed.")
    logger.info("Script finished.")