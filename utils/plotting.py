# utils/plotting.py
import matplotlib.pyplot as plt
import torch

def plot_spectrograms_to_figure(ground_truth, prediction, title="Spectrogram Comparison", vmin=-80, vmax=0):
    """
    Generates a matplotlib figure comparing ground truth and predicted spectrograms.

    Args:
        ground_truth (torch.Tensor): Ground truth mel-spectrogram (Freq, Time).
        prediction (torch.Tensor): Predicted mel-spectrogram (Freq, Time).
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)

    # Ensure tensors are on CPU and detached for plotting
    gt_np = ground_truth.cpu().detach().numpy()
    pred_np = prediction.cpu().detach().numpy()

    # Plot Ground Truth
    im_gt = axes[0].imshow(gt_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_ylabel("Mel Bin")
    cbar_gt = fig.colorbar(im_gt, ax=axes[0])
    # cbar_gt.set_label('dB') # Removed dB label

    # Plot Prediction
    im_pred = axes[1].imshow(pred_np, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")
    cbar_pred = fig.colorbar(im_pred, ax=axes[1])
    # cbar_pred.set_label('dB') # Removed dB label

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig