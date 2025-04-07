# utils/plotting.py
import matplotlib.pyplot as plt
import torch
import numpy as np # Added numpy import
from typing import Optional # Added for type hinting

def plot_spectrograms_to_figure(ground_truth: torch.Tensor, prediction: torch.Tensor, title: str = "Spectrogram Comparison", vmin: float = -80, vmax: float = 4, cmap: str = 'viridis') -> plt.Figure: # Set user preferred defaults
    """
    Generates a matplotlib figure comparing ground truth and predicted spectrograms.

    Args:
        ground_truth (torch.Tensor): Ground truth mel-spectrogram (Freq, Time).
        prediction (torch.Tensor): Predicted mel-spectrogram (Freq, Time).
        title (str): Title for the plot.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
        cmap (str): Colormap to use for the plots.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)

    # Ensure tensors are on CPU and detached for plotting
    gt_np = ground_truth.cpu().detach().numpy()
    pred_np = prediction.cpu().detach().numpy()

    # Plot Ground Truth
    im_gt = axes[0].imshow(gt_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap) # Added cmap
    axes[0].set_title("Ground Truth")
    axes[0].set_ylabel("Mel Bin")
    cbar_gt = fig.colorbar(im_gt, ax=axes[0])
    # cbar_gt.set_label('dB') # Removed dB label

    # Plot Prediction
    im_pred = axes[1].imshow(pred_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap) # Added cmap
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")
    cbar_pred = fig.colorbar(im_pred, ax=axes[1])
    # cbar_pred.set_label('dB') # Removed dB label

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig


def plot_single_spectrogram_to_figure(prediction: torch.Tensor, title: str = "Spectrogram", vmin: float = -80, vmax: float = 0) -> plt.Figure:
    """
    Generates a matplotlib figure displaying a single spectrogram.

    Args:
        prediction (torch.Tensor): Predicted mel-spectrogram (Freq, Time).
        title (str): Title for the plot.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4)) # Single subplot
    fig.suptitle(title)

    # Ensure tensor is on CPU and detached for plotting
    pred_np = prediction.cpu().detach().numpy()

    # Plot Prediction
    im_pred = ax.imshow(pred_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title("Prediction")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel Bin")
    cbar_pred = fig.colorbar(im_pred, ax=ax)
    # cbar_pred.set_label('dB') # Removed dB label

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    return fig


def plot_waveforms_to_figure(ground_truth: torch.Tensor, prediction: torch.Tensor, sample_rate: int, title: str = "Waveform Comparison") -> plt.Figure:
   """
   Generates a matplotlib figure comparing ground truth and predicted waveforms overlaid.

   Args:
       ground_truth (torch.Tensor): Ground truth waveform (1D tensor).
       prediction (torch.Tensor): Predicted waveform (1D tensor).
       sample_rate (int): Sample rate of the audio.
       title (str): Title for the plot.

   Returns:
       matplotlib.figure.Figure: The generated figure.
   """
   fig, ax = plt.subplots(1, 1, figsize=(12, 4)) # Single subplot
   fig.suptitle(title)

   # Ensure tensors are on CPU and detached for plotting
   gt_np = ground_truth.cpu().detach().numpy()
   pred_np = prediction.cpu().detach().numpy()

   # Create time axis
   duration = len(gt_np) / sample_rate
   time_axis = np.linspace(0, duration, num=len(gt_np))

   # Plot waveforms
   ax.plot(time_axis, gt_np, color='blue', label='Ground Truth', alpha=0.8)
   ax.plot(time_axis, pred_np, color='red', label='Prediction', alpha=0.8)

   # Add labels and legend
   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Amplitude")
   ax.legend()
   ax.grid(True, linestyle='--', alpha=0.6)
   ax.set_xlim(0, duration) # Ensure x-axis covers the full duration

   plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
   return fig

# --- NEW FUNCTION ---
def plot_spectrograms_comparison_to_figure(
    spec1: torch.Tensor,
    spec2: torch.Tensor,
    title1: str = "Spectrogram 1",
    title2: str = "Spectrogram 2",
    main_title: str = "Spectrogram Comparison",
    vmin: float = -80,
    vmax: float = 4,
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Generates a matplotlib figure comparing two spectrograms side-by-side.

    Args:
        spec1 (torch.Tensor): First spectrogram (Freq, Time).
        spec2 (torch.Tensor): Second spectrogram (Freq, Time).
        title1 (str): Title for the first spectrogram plot.
        title2 (str): Title for the second spectrogram plot.
        main_title (str): Overall title for the figure.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
        cmap (str): Colormap to use for the plots.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8)) # Keep vertical layout for better time comparison
    fig.suptitle(main_title)

    # Ensure tensors are on CPU and detached
    spec1_np = spec1.cpu().detach().numpy()
    spec2_np = spec2.cpu().detach().numpy()

    # Plot Spectrogram 1
    im1 = axes[0].imshow(spec1_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(f"{title1} (Time: {spec1_np.shape[1]})") # Add time dim to title
    axes[0].set_ylabel("Mel Bin")
    fig.colorbar(im1, ax=axes[0])

    # Plot Spectrogram 2
    im2 = axes[1].imshow(spec2_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(f"{title2} (Time: {spec2_np.shape[1]})") # Add time dim to title
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
# --- END NEW FUNCTION ---