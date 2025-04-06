# inference.py
import torch
import yaml
import argparse
import os
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
import warnings

# Filter specific UserWarning from h5py
warnings.filterwarnings("ignore", message=r"h5py is running against HDF5.*when it was built against.*", category=UserWarning)

# Import custom modules
from data.dataset import DataModule, collate_fn_pad, H5FileManager # Import H5FileManager
from models.progressive_svs import ProgressiveSVS
from utils.plotting import plot_spectrograms_to_figure

def main(args):
    """Main inference function."""
    print(f"Loading configuration from: {args.config_path}")
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return

    # --- Inject Stage into Config ---
    config['model']['current_stage'] = args.stage
    print(f"Set inference stage to: {args.stage}")

    # --- Determine Device ---
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Initializing {config['model']['name']} model for stage {args.stage}...")
    # We need vocab_size for model init, get it from DataModule setup later
    # Temporarily set vocab_size to None or a placeholder if needed for init
    # config['model']['vocab_size'] = config['model'].get('vocab_size') # Use value from config if present
    model = ProgressiveSVS(config) # Initialize with potentially incomplete config

    print(f"Loading checkpoint from: {args.ckpt_path}")
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint file not found at {args.ckpt_path}")
        return
    try:
        # Load full checkpoint dictionary
        checkpoint = torch.load(args.ckpt_path, map_location='cpu') # Load to CPU first

        # --- Update Config with Checkpoint Hyperparameters ---
        # This ensures model parameters match the checkpoint
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
             print("Updating config with hyperparameters from checkpoint...")
             # Carefully merge checkpoint config into loaded config
             # Prioritize checkpoint values for model architecture params
             # Example: update model section, but keep data paths from file config
             checkpoint_config = checkpoint['hyper_parameters']['config']
             # Update model section
             if 'model' in checkpoint_config:
                 config['model'].update(checkpoint_config['model'])
             # Update other relevant sections if necessary, be cautious
             # config['train'].update(checkpoint_config.get('train', {})) # Example if needed
             print(f"Using vocab_size from checkpoint: {config['model'].get('vocab_size')}")
             # Re-initialize model if config changed significantly (optional, usually load_state_dict handles it)
             # model = ProgressiveSVS(config) # Re-init if needed
        else:
             print("Warning: Checkpoint does not contain hyperparameter config. Using config file values.")
             # Ensure vocab_size is handled later by DataModule if not in config/checkpoint

        # Load Model Weights
        print("Loading model state_dict...")
        # Extract state_dict - handle potential nested structure if saved by Lightning >= 2.0
        state_dict = checkpoint.get('state_dict', checkpoint) # Check for 'state_dict' key first
        # Remove potential 'model.' prefix if saved with DataParallel or DistributedDataParallel
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        # Load weights, allow missing/unexpected keys if architecture changed slightly
        model.load_state_dict(state_dict, strict=False)
        print("Model state_dict loaded.")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- Load Data ---
    print("Initializing DataModule...")
    try:
        data_module = DataModule(config)
        data_module.setup() # Prepare dataset and read vocab_size if not in checkpoint
        # Ensure model's vocab_size matches dataset's
        if config['model'].get('vocab_size') is None:
             if data_module.vocab_size is not None:
                 print(f"Setting model vocab_size from DataModule: {data_module.vocab_size}")
                 config['model']['vocab_size'] = data_module.vocab_size
                 # If model wasn't initialized with correct vocab_size, might need re-init or careful weight loading
             else:
                 print("Error: vocab_size not found in checkpoint, config, or dataset attributes.")
                 return
        elif data_module.vocab_size is not None and config['model']['vocab_size'] != data_module.vocab_size:
             print(f"Warning: vocab_size mismatch! Checkpoint/Config: {config['model']['vocab_size']}, Dataset: {data_module.vocab_size}. Using Checkpoint/Config value.")
             # Potentially update embedding layer size if necessary and possible

        # --- Select Dataset based on flag ---
        if args.use_val_split:
            if hasattr(data_module, 'val_dataset') and data_module.val_dataset is not None:
                dataset = data_module.val_dataset
                print(f"Using VALIDATION dataset split with {len(dataset)} samples.")
            else:
                print("Error: --use_val_split specified, but validation dataset is not available in DataModule.")
                print("Check if 'validation_split' in config was > 0 during training setup.")
                return
        else:
            dataset = data_module.dataset
            print(f"Using FULL dataset with {len(dataset)} samples.")
        # --- End Dataset Selection ---
    except FileNotFoundError as e:
         h5_path_expected = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
         print(f"Error: HDF5 dataset file not found at '{h5_path_expected}'. Did you run preprocessing?")
         print(f"Original error: {e}")
         return
    except ValueError as e: # Catch the ValueError raised if vocab_size attr is missing
        print(f"Error during DataModule setup: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during DataModule initialization/setup: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Select Sample ---
    if args.sample_index is not None:
        if 0 <= args.sample_index < len(dataset):
            sample_idx = args.sample_index
            print(f"Using specified sample index: {sample_idx}")
        else:
            print(f"Error: Invalid sample_index {args.sample_index}. Must be between 0 and {len(dataset)-1}.")
            return
    else:
        sample_idx = random.randint(0, len(dataset) - 1)
        print(f"Using random sample index: {sample_idx}")

    try:
        sample = dataset[sample_idx]
        if sample is None:
             print(f"Error: Failed to load sample at index {sample_idx}.")
             # Close HDF5 file manager if necessary
             H5FileManager.get_instance().close_all()
             return
    except Exception as e:
        print(f"Error retrieving sample {sample_idx}: {e}")
        # Close HDF5 file manager if necessary
        H5FileManager.get_instance().close_all()
        return

    # --- Prepare Batch ---
    print("Preparing batch...")
    batch = [sample]
    collated_batch = collate_fn_pad(batch)
    if collated_batch is None:
        print("Error: Collate function returned None. Could not process sample.")
        # Close HDF5 file manager if necessary
        H5FileManager.get_instance().close_all()
        return

    # Move batch to device
    for key, tensor in collated_batch.items():
        if isinstance(tensor, torch.Tensor):
            collated_batch[key] = tensor.to(device)
    print("Batch prepared and moved to device.")

    # --- Inference ---
    print("Running inference...")
    with torch.no_grad():
        # Extract required inputs
        f0 = collated_batch['f0']
        phone_label = collated_batch['phone_label']
        phone_duration = collated_batch['phone_duration']
        midi_label = collated_batch['midi_label']
        unvoiced_flag = collated_batch['unvoiced_flag'] # Added unvoiced_flag

        # Run model forward pass
        predicted_mel = model(f0, phone_label, phone_duration, midi_label, unvoiced_flag)
    print("Inference complete.")

    # --- Prepare Ground Truth for Comparison ---
    print("Preparing ground truth spectrogram...")
    mel_spec_gt_orig = collated_batch['mel_spec'] # Ground truth from batch [B, T_orig, F_orig]
    original_max_len = mel_spec_gt_orig.shape[1]
    target_time_dim = original_max_len // model.downsample_stride # Use stride from model

    # Replicate downsampling logic from validation_step
    if args.stage == 1:
        scale_factor = 1 / config['model']['low_res_scale']
        freq_dim = int(config['model']['mel_bins'] * scale_factor)
        mel_specs_permuted = mel_spec_gt_orig.permute(0, 2, 1) # B, F, T
        mel_specs_4d = mel_specs_permuted.unsqueeze(1) # B, 1, F, T
        target_mel_downsampled = F.interpolate(
            mel_specs_4d, size=(freq_dim, target_time_dim), mode='bilinear', align_corners=False
        ).squeeze(1) # B, F', T'
        target_mel = target_mel_downsampled.permute(0, 2, 1) # B, T', F'
    elif args.stage == 2:
        scale_factor = 1 / config['model']['mid_res_scale']
        freq_dim = int(config['model']['mel_bins'] * scale_factor)
        mel_specs_permuted = mel_spec_gt_orig.permute(0, 2, 1) # B, F, T
        mel_specs_4d = mel_specs_permuted.unsqueeze(1) # B, 1, F, T
        target_mel_downsampled = F.interpolate(
            mel_specs_4d, size=(freq_dim, target_time_dim), mode='nearest' # Use nearest like in validation
        ).squeeze(1) # B, F', T'
        target_mel = target_mel_downsampled.permute(0, 2, 1) # B, T', F'
    else: # Stage 3
        freq_dim = config['model']['mel_bins']
        mel_specs_permuted = mel_spec_gt_orig.permute(0, 2, 1) # B, F, T
        mel_specs_4d = mel_specs_permuted.unsqueeze(1) # B, 1, F, T
        target_mel_downsampled = F.interpolate(
            mel_specs_4d, size=(freq_dim, target_time_dim), mode='bilinear', align_corners=False
        ).squeeze(1) # B, F, T'
        target_mel = target_mel_downsampled.permute(0, 2, 1) # B, T', F

    # Ensure prediction and target have same shape
    if predicted_mel.shape != target_mel.shape:
         print(f"Warning: Shape mismatch after GT prep. Pred: {predicted_mel.shape}, Target: {target_mel.shape}. Trying to adjust prediction.")
         # Example adjustment (might need refinement based on actual mismatch)
         if predicted_mel.dim() == 4 and predicted_mel.size(1) == 1:
             predicted_mel = predicted_mel.squeeze(1).permute(0, 2, 1)
         # Add more checks if needed
         if predicted_mel.shape != target_mel.shape:
              print("Error: Could not resolve shape mismatch between prediction and target.")
              # Close HDF5 file manager if necessary
              H5FileManager.get_instance().close_all()
              return

    # --- Visualization ---
    print("Generating visualization...")
    # Get the first (and only) item from the batch
    pred_mel_full = predicted_mel[0] # Shape (T_padded', F')
    target_mel_full = target_mel[0] # Shape (T_padded', F')

    # Get the original length and calculate downsampled length
    original_length = collated_batch['length'][0].item() # Get length for the first item
    downsampled_length = original_length // model.downsample_stride
    print(f"Original length: {original_length}, Downsampled length for plotting: {downsampled_length}")

    # Slice the tensors to the actual length
    pred_mel_sliced = pred_mel_full[:downsampled_length, :]
    target_mel_sliced = target_mel_full[:downsampled_length, :]

    # Transpose for plotting (Freq, Time)
    pred_mel_plot = pred_mel_sliced.T
    target_mel_plot = target_mel_sliced.T

    # vmin/vmax will be calculated dynamically below based on mode

    if args.mode == 'analysis':
        print("Generating analysis plot (Prediction vs Ground Truth)...")
        # Calculate vmin/vmax dynamically from the combined range of sliced GT and prediction
        vmin = min(target_mel_sliced.min().item(), pred_mel_sliced.min().item())
        vmax = max(target_mel_sliced.max().item(), pred_mel_sliced.max().item())
        print(f"Using combined dynamic vmin={vmin:.4f}, vmax={vmax:.4f}")
        fig = plot_spectrograms_to_figure(
            target_mel_plot, # Already transposed
            pred_mel_plot,   # Already transposed
            title=f"Analysis Stage {args.stage} - Sample {sample_idx}",
            vmin=vmin,
            vmax=vmax
        )
    elif args.mode == 'synthesis':
        print("Generating synthesis plot (Prediction only)...")
        # Calculate vmin/vmax dynamically from the sliced prediction
        vmin = pred_mel_sliced.min().item()
        vmax = pred_mel_sliced.max().item()
        print(f"Using dynamic vmin={vmin:.4f}, vmax={vmax:.4f} from prediction")
        fig, ax = plt.subplots(1, 1, figsize=(10, 4)) # Single subplot
        im = ax.imshow(
            pred_mel_plot.cpu().detach().numpy(), # Use transposed, sliced prediction
            aspect='auto',
            origin='lower',
            interpolation='none',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"Synthesis Stage {args.stage} - Sample {sample_idx}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel Bin")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
    else:
        # Should not happen due to argparse choices, but handle defensively
        print(f"Error: Unknown mode '{args.mode}'")
        H5FileManager.get_instance().close_all()
        return

    # --- Save Output ---
    try:
        output_dir = os.path.dirname(args.output_image)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(args.output_image)
        plt.close(fig)
        print(f"Output visualization saved to: {args.output_image}")
    except Exception as e:
        print(f"Error saving output image: {e}")

    # --- Cleanup ---
    # Close HDF5 files managed by the singleton
    H5FileManager.get_instance().close_all()
    print("HDF5 files closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained ProgressiveSVS model.")
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='The training stage the checkpoint corresponds to (1, 2, or 3).')
    parser.add_argument('--config_path', type=str, default='config/model.yaml',
                        help='Path to the configuration YAML file (default: config/model.yaml).')
    parser.add_argument('--output_image', type=str, default='inference_output.png',
                        help='File path to save the output visualization (default: inference_output.png).')
    parser.add_argument('--sample_index', type=int, default=None,
                        help='Specific index of the dataset sample to use (default: random).')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help="Device to run inference on ('cpu', 'cuda', 'auto'; default: auto).")
    parser.add_argument('--mode', type=str, default='analysis', choices=['analysis', 'synthesis'],
                        help="Inference mode: 'analysis' (compare GT and prediction) or 'synthesis' (generate prediction only) (default: analysis).")
    parser.add_argument('--use_val_split', action='store_true',
                        help="If specified, load and select samples from the validation dataset split instead of the full dataset.")

    args = parser.parse_args()

    # Set seed for reproducibility of random sample selection if desired
    pl.seed_everything(42, workers=True)

    main(args)