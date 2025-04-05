# visualize_data.py
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def print_h5_structure(h5_file, prefix=''):
    """Recursively prints the structure of an HDF5 file."""
    for key in h5_file.keys():
        item = h5_file[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            print(f"{path} (Dataset): Shape={item.shape}, Dtype={item.dtype}")
            # Print attributes of the dataset
            if item.attrs:
                print(f"  Attributes:")
                for attr_name, attr_val in item.attrs.items():
                    print(f"    - {attr_name}: {attr_val}")
        elif isinstance(item, h5py.Group):
            print(f"{path} (Group)")
            # Recursively print structure of the group
            print_h5_structure(item, prefix=path)

def visualize_sample(h5_file, index, data_key):
    """Visualizes a single sample from the HDF5 file."""
    print(f"\nVisualizing sample index: {index}")

    # --- Access Datasets ---
    try:
        mel_dataset = h5_file[data_key]
    except KeyError:
        print(f"Error: Data key '{data_key}' not found in the H5 file.")
        return

    lengths_dataset = h5_file.get('lengths')
    file_ids_dataset = h5_file.get('file_ids')
    f0_dataset = h5_file.get('F0') # Assuming F0 is stored directly under root

    # --- Load Data for the Index ---
    try:
        mel_spec = mel_dataset[index]
    except IndexError:
        print(f"Error: Index {index} is out of bounds for dataset '{data_key}' (size: {len(mel_dataset)}).")
        return

    actual_length = None
    if lengths_dataset is not None and index < len(lengths_dataset):
        actual_length = lengths_dataset[index]
        print(f"  Actual length (frames): {actual_length}")
        # Trim padding if actual_length is available and less than total frames
        if actual_length < mel_spec.shape[1]:
             mel_spec = mel_spec[:, :actual_length]
    else:
        print("  'lengths' dataset not found or index out of bounds.")
        actual_length = mel_spec.shape[1] # Use the full length

    file_id = None
    if file_ids_dataset is not None and index < len(file_ids_dataset):
        # Decode if bytes
        file_id = file_ids_dataset[index]
        if isinstance(file_id, bytes):
            try:
                file_id = file_id.decode('utf-8')
            except UnicodeDecodeError:
                print("  Warning: Could not decode file_id.")
                file_id = str(file_id) # Keep as string representation
        print(f"  File ID: {file_id}")
    else:
        print("  'file_ids' dataset not found or index out of bounds.")

    f0_data = None
    if f0_dataset is not None and index < len(f0_dataset):
        f0_data = f0_dataset[index]
        # Trim F0 data to match actual mel spectrogram length
        if f0_data is not None and len(f0_data) > actual_length:
            f0_data = f0_data[:actual_length]
        print(f"  F0 data shape: {f0_data.shape if f0_data is not None else 'Not found'}")
    else:
        print("  'F0' dataset not found or index out of bounds.")

    # --- Read Audio Attributes ---
    sample_rate = mel_dataset.attrs.get('sample_rate', None)
    hop_length = mel_dataset.attrs.get('hop_length', None)
    n_mels = mel_dataset.attrs.get('n_mels', mel_spec.shape[0]) # Fallback to shape
    print(f"  Audio Attributes: SR={sample_rate}, Hop={hop_length}, MelBins={n_mels}")

    # --- Create Plot ---
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Display Mel Spectrogram
    img = ax1.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(img, ax=ax1, format='%+2.0f dB') # Assuming mel specs are in dB scale
    ax1.set_ylabel(f'Mel Bins ({n_mels})')
    ax1.set_xlabel('Time Frames')

    # Prepare title
    title = f'Sample Index: {index}'
    if file_id:
        title += f' | File ID: {file_id}'
    if actual_length:
         title += f' | Length: {actual_length} frames'
    if sample_rate and hop_length:
        duration_sec = actual_length * hop_length / sample_rate
        title += f' ({duration_sec:.2f}s)'
        # Optionally set x-axis to seconds
        time_axis = np.arange(actual_length) * hop_length / sample_rate
        # ax1.set_xticks(np.linspace(0, actual_length - 1, num=5)) # Keep frames for now
        # ax1.set_xticklabels([f"{t:.2f}" for t in np.linspace(0, duration_sec, num=5)])
        # ax1.set_xlabel('Time (s)')


    # Plot F0 on secondary axis if available
    if f0_data is not None and len(f0_data) == actual_length:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('F0 (Hz)', color=color)
        ax2.plot(np.arange(actual_length), f0_data, color=color, linewidth=1.5, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0) # F0 shouldn't be negative
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    else:
        if f0_data is not None:
            print(f"  Warning: F0 data length ({len(f0_data)}) does not match Mel length ({actual_length}). Skipping F0 plot.")

    ax1.set_title(title)
    plt.show(block=False) # Use block=False to allow multiple plots

def main():
    parser = argparse.ArgumentParser(description='Visualize samples from an HDF5 dataset.')
    parser.add_argument('--h5_path', type=str, default='datasets/mel_spectrograms.h5',
                        help='Path to the HDF5 dataset file.')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms',
                        help='Key for the main data (e.g., mel spectrograms) in the HDF5 file.')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to visualize.')
    parser.add_argument('--index', type=int, default=None,
                        help='Specific index of a sample to visualize (overrides --num_samples).')
    parser.add_argument('--info', action='store_true',
                        help='Print information about the HDF5 file structure and exit.')

    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"Error: HDF5 file not found at {args.h5_path}")
        return

    try:
        with h5py.File(args.h5_path, 'r') as h5_file:
            print(f"Opened HDF5 file: {args.h5_path}")

            if args.info:
                print("\nHDF5 File Structure:")
                print("--------------------")
                print_h5_structure(h5_file)
                print("--------------------")
                return # Exit after printing info

            # --- Determine indices to visualize ---
            try:
                total_samples = len(h5_file[args.data_key])
            except KeyError:
                 print(f"Error: Data key '{args.data_key}' not found in the H5 file.")
                 return
            except Exception as e:
                 print(f"Error accessing dataset '{args.data_key}': {e}")
                 return

            if total_samples == 0:
                print(f"Dataset '{args.data_key}' is empty.")
                return

            indices_to_visualize = []
            if args.index is not None:
                if 0 <= args.index < total_samples:
                    indices_to_visualize.append(args.index)
                else:
                    print(f"Error: Provided index {args.index} is out of bounds (0-{total_samples-1}).")
                    return
            else:
                num_to_show = min(args.num_samples, total_samples)
                if num_to_show > 0:
                    indices_to_visualize = random.sample(range(total_samples), num_to_show)
                print(f"Selecting {num_to_show} random samples out of {total_samples}.")

            if not indices_to_visualize:
                print("No samples selected for visualization.")
                return

            # --- Visualize selected samples ---
            for idx in indices_to_visualize:
                visualize_sample(h5_file, idx, args.data_key)

            # Keep plots open until user closes them
            print("\nClose plot windows to exit.")
            plt.show() # This will block until all figures are closed

    except FileNotFoundError:
        print(f"Error: File not found at {args.h5_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()