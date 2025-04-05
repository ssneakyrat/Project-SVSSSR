# visualize_data.py
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def print_h5_structure(h5_file, prefix=''):
    """Recursively prints the structure of an HDF5 file, including attributes."""
    # Print attributes of the current level (file or group)
    if h5_file.attrs:
        print(f"{prefix}/ (Attributes):")
        for attr_name, attr_val in h5_file.attrs.items():
            # Truncate long attribute values for display
            attr_val_str = str(attr_val)
            if len(attr_val_str) > 100:
                attr_val_str = attr_val_str[:100] + "..."
            print(f"  - {attr_name}: {attr_val_str}")

    # Iterate through keys (datasets and subgroups)
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

def visualize_sample(group, id_to_phone_map):
    """Visualizes the data within a specific group (filename) of the HDF5 file."""
    filename = group.name.lstrip('/')
    print(f"\nVisualizing data for: {filename}")

    try:
        # --- Load Data from Group ---
        # Use .get() for robustness, provide default empty arrays if missing
        mel = group.get('mel_spectrograms')[:] if 'mel_spectrograms' in group else np.array([])
        f0 = group.get('f0')[:] if 'f0' in group else np.array([])
        phone_ids = group.get('phone_sequence')[:] if 'phone_sequence' in group else np.array([])
        # Use initial durations for visualization as requested
        durations = group.get('initial_duration_sequence')[:] if 'initial_duration_sequence' in group else np.array([])

        if mel.size == 0:
            print(f"  Error: 'mel_spectrograms' dataset not found or empty in group '{filename}'.")
            return
        if f0.size == 0:
            print(f"  Warning: 'f0' dataset not found or empty in group '{filename}'.")
        if phone_ids.size == 0:
            print(f"  Error: 'phone_sequence' dataset not found or empty in group '{filename}'.")
            return
        if durations.size == 0:
            print(f"  Error: 'initial_duration_sequence' dataset not found or empty in group '{filename}'.")
            return
        if len(phone_ids) != len(durations):
             print(f"  Error: Mismatch between length of 'phone_sequence' ({len(phone_ids)}) and 'initial_duration_sequence' ({len(durations)}).")
             return

        print(f"  Mel shape: {mel.shape}")
        print(f"  F0 shape: {f0.shape}")
        print(f"  Phone sequence length: {len(phone_ids)}")
        print(f"  Initial duration sequence length: {len(durations)}")

        # --- Create Plot (3 subplots) ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Data for: {filename}", fontsize=14)

        # 1. Mel Spectrogram
        img = axes[0].imshow(mel.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_ylabel('Mel Bin')
        axes[0].set_title('Mel Spectrogram')
        # Removed colorbar as per previous request

        # 2. F0 Contour
        # Ensure F0 length matches Mel length (it should if preprocessing worked)
        mel_frames = mel.shape[0]
        if len(f0) != mel_frames:
             print(f"  Warning: F0 length ({len(f0)}) differs from Mel length ({mel_frames}). Truncating/Padding F0 for plot.")
             if len(f0) > mel_frames:
                 f0 = f0[:mel_frames]
             else:
                 f0 = np.pad(f0, ((0, mel_frames - len(f0)), (0,0)), mode='constant') # Pad along time axis

        axes[1].plot(f0, label='F0')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_title('F0 Contour')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # 3. Phone Alignment (using initial durations)
        axes[2].set_title('Phone Alignment (Initial Scaled Durations)')
        axes[2].set_ylabel('Phone ID')
        axes[2].set_xlabel('Frame')
        axes[2].grid(True, linestyle='--', alpha=0.6)

        current_frame = 0
        yticks = []
        phone_boundaries = [0]
        total_initial_duration = sum(durations)

        for phone_id, duration in zip(phone_ids, durations):
            phone_symbol = id_to_phone_map.get(str(phone_id), '?') # Map keys might be strings if loaded from JSON
            center_frame = current_frame + duration / 2
            # Plot text only if duration is > 0
            if duration > 0:
                 axes[2].text(center_frame, phone_id, phone_symbol, ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            current_frame += duration
            phone_boundaries.append(current_frame)
            if phone_id not in yticks:
                 yticks.append(phone_id)

        # Draw vertical lines for phone boundaries
        for boundary in phone_boundaries:
             axes[2].axvline(x=boundary, color='r', linestyle='--', linewidth=0.8)

        # Set y-ticks for phones
        if yticks:
            # Ensure map keys used for lookup are strings if necessary
            axes[2].set_yticks(sorted(list(set(yticks))))
            axes[2].set_yticklabels([id_to_phone_map.get(str(y), '?') for y in sorted(list(set(yticks)))])
            axes[2].set_ylim(min(yticks)-1 if yticks else -1, max(yticks)+1 if yticks else 1)
        else:
             axes[2].set_yticks([]) # No phones, clear ticks

        # Adjust x-axis limit to match the total initial duration
        axes[2].set_xlim(0, total_initial_duration)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show(block=False)

    except Exception as e:
        print(f"An error occurred during visualization for {filename}: {e}")
        import traceback
        traceback.print_exc()

import json # Add json import

def main():
    parser = argparse.ArgumentParser(description='Visualize data for a specific file within an HDF5 dataset.')
    parser.add_argument('--h5_path', type=str, default='datasets/mel_spectrograms.h5',
                        help='Path to the HDF5 dataset file.')
    parser.add_argument('--filename', type=str, default=None,
                        help='Filename (group name) inside the HDF5 file to visualize. If omitted, lists available files.')
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

            # --- Load Phone Map ---
            id_to_phone_map = None
            try:
                if 'id_to_phone_map' in h5_file.attrs:
                    id_to_phone_map = json.loads(h5_file.attrs['id_to_phone_map'])
                    # Convert keys back to int if needed, though string keys work fine for dict.get()
                    # id_to_phone_map = {int(k): v for k, v in id_to_phone_map.items()}
                else:
                    print("Error: 'id_to_phone_map' attribute not found in HDF5 file root.")
                    return
            except json.JSONDecodeError:
                print("Error: Failed to decode 'id_to_phone_map' attribute from JSON.")
                return
            except Exception as e:
                 print(f"Error loading id_to_phone_map: {e}")
                 return

            # --- Select file/group to visualize ---
            target_group = None
            if args.filename:
                if args.filename in h5_file:
                    target_group = h5_file[args.filename]
                    if not isinstance(target_group, h5py.Group):
                         print(f"Error: '{args.filename}' exists but is not a Group.")
                         return
                else:
                    print(f"Error: Filename (group) '{args.filename}' not found in the HDF5 file.")
                    print("\nAvailable files:")
                    for name in h5_file:
                        if isinstance(h5_file[name], h5py.Group):
                            print(f"- {name}")
                    return
            else:
                # List available files and exit if no specific file requested
                print("\nNo specific filename provided. Available files (groups):")
                found_groups = False
                for name in h5_file:
                     if isinstance(h5_file[name], h5py.Group):
                         print(f"- {name}")
                         found_groups = True
                if not found_groups:
                    print("(No groups found)")
                print("\nUse the --filename argument to specify which file to visualize.")
                return

            # --- Visualize the selected group ---
            if target_group:
                visualize_sample(target_group, id_to_phone_map)
                # Keep plot open until user closes it
                print("\nClose plot window to exit.")
                plt.show() # This will block until the figure is closed
            else:
                # This case should ideally be handled above, but as a fallback:
                print("No target file selected for visualization.")


    except FileNotFoundError:
        print(f"Error: File not found at {args.h5_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()