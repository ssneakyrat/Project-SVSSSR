# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude and phase."""
    x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window,
                        return_complex=True, normalized=False, center=True, pad_mode='reflect')
    # Output shape: [B, F, T] complex
    # Separate magnitude and phase is often not needed directly for loss,
    # but magnitude is used. Complex tensor is fine for calculations.
    return x_stft # Return complex tensor

class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Predicted STFT magnitude map (B, #frames, #freq_bins).
            y_mag (Tensor): Ground-truth STFT magnitude map (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        # Ensure non-negative magnitudes before norm calculation
        x_mag = torch.clamp(x_mag, min=0.0)
        y_mag = torch.clamp(y_mag, min=0.0)

        # Calculate the Frobenius norm of the difference and the ground truth magnitude
        norm_diff = torch.norm(y_mag - x_mag, p='fro', dim=(-2, -1)) # Norm across Freq and Time dims
        norm_y = torch.norm(y_mag, p='fro', dim=(-2, -1))

        # Prevent division by zero
        loss_sc = torch.mean(norm_diff / norm_y.clamp(min=1e-8)) # Average over batch

        return loss_sc

class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Predicted STFT magnitude map (B, #frames, #freq_bins).
            y_mag (Tensor): Ground-truth STFT magnitude map (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        # Ensure non-negative magnitudes before log
        x_mag = torch.clamp(x_mag, min=1e-7) # Add small epsilon for log stability
        y_mag = torch.clamp(y_mag, min=1e-7)

        # Calculate L1 loss on the log magnitudes
        loss_mag = F.l1_loss(torch.log(y_mag), torch.log(x_mag))

        return loss_mag


class STFTLoss(nn.Module):
    """STFT loss module."""
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240], window="hann_window",
                 loss_sc_weight=1.0, loss_mag_weight=1.0):
        """Initialize STFT loss module."""
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        # Register windows as buffers
        for i, (fft_size, win_length) in enumerate(zip(fft_sizes, win_lengths)):
             assert win_length <= fft_size, f"win_length {win_length} must be <= fft_size {fft_size}"
             # Get window tensor and register it
             window_fn = getattr(torch, window)
             win = window_fn(win_length, periodic=False) # Use periodic=False for Hann
             self.register_buffer(f"window_{i}", win)

        self.loss_sc_weight = loss_sc_weight
        self.loss_mag_weight = loss_mag_weight
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        logger.info(f"Initialized STFTLoss with FFT sizes: {fft_sizes}, Hops: {hop_sizes}, Wins: {win_lengths}")
        logger.info(f"STFTLoss weights: SC={loss_sc_weight}, Mag={loss_mag_weight}")


    def forward(self, y_pred, y_true, lengths=None):
        """Calculate forward propagation.
        Args:
            y_pred (Tensor): Predicted waveform tensor (B, T).
            y_true (Tensor): Ground-truth waveform tensor (B, T).
            lengths (Tensor, optional): Lengths of the waveforms (B,). Used for masking if needed. Defaults to None.

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
            Tensor: Combined STFT loss value.
        """
        loss_sc_total = 0.0
        loss_mag_total = 0.0

        # Ensure inputs are 1D or 2D (B, T)
        if y_pred.dim() > 2 or y_true.dim() > 2:
             logger.warning(f"STFTLoss input dims > 2. y_pred: {y_pred.shape}, y_true: {y_true.shape}. Squeezing.")
             y_pred = y_pred.squeeze()
             y_true = y_true.squeeze()
        if y_pred.dim() == 1: # Add batch dim if single sample
             y_pred = y_pred.unsqueeze(0)
             y_true = y_true.unsqueeze(0)

        # Ensure lengths match batch size if provided
        if lengths is not None and len(lengths) != y_pred.shape[0]:
             logger.error(f"Length tensor size ({len(lengths)}) does not match batch size ({y_pred.shape[0]})")
             lengths = None # Ignore lengths if mismatch

        for i, (fft_size, hop_size, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"window_{i}")
            window = window.to(y_pred.device) # Ensure window is on the correct device

            # Calculate STFT for predicted and true waveforms
            stft_true_complex = stft(y_true, fft_size, hop_size, win_length, window) # [B, F, T_spec]
            stft_pred_complex = stft(y_pred, fft_size, hop_size, win_length, window) # [B, F, T_spec]

            # Get magnitudes
            stft_true_mag = torch.abs(stft_true_complex) # [B, F, T_spec]
            stft_pred_mag = torch.abs(stft_pred_complex) # [B, F, T_spec]

            # --- Optional Masking (if lengths are provided) ---
            # This is complex as STFT changes time dimension.
            # A simple approach is to mask the loss calculation based on original lengths,
            # but it's often omitted for simplicity, assuming padding doesn't dominate loss.
            # If masking is needed, calculate the mask for the STFT time dimension based on `lengths` and `hop_size`.
            # mask = ... # Shape [B, 1, T_spec]
            # Apply mask before loss calculation, e.g., F.l1_loss(log(y_mag*mask), log(x_mag*mask))
            # and adjust norm calculation accordingly.
            # For now, we omit masking for simplicity.
            # --- End Optional Masking ---


            # Calculate losses for this resolution
            # Note: Loss functions expect [B, F, T] or similar, transpose if needed by specific loss impl.
            # Our current impl handles [B, F, T] directly.
            loss_sc = self.spectral_convergence_loss(stft_pred_mag, stft_true_mag)
            loss_mag = self.log_stft_magnitude_loss(stft_pred_mag, stft_true_mag)

            loss_sc_total += loss_sc
            loss_mag_total += loss_mag

        # Average losses over the number of resolutions
        loss_sc_avg = loss_sc_total / len(self.fft_sizes)
        loss_mag_avg = loss_mag_total / len(self.fft_sizes)

        # Combine losses with weights
        loss_total = (self.loss_sc_weight * loss_sc_avg) + (self.loss_mag_weight * loss_mag_avg)

        return loss_sc_avg, loss_mag_avg, loss_total

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Example parameters matching default STFTLoss
    fft_sizes = [1024, 2048, 512]
    hop_sizes = [120, 240, 50]
    win_lengths = [600, 1200, 240]
    loss_sc_weight = 1.0
    loss_mag_weight = 1.0

    stft_loss_fn = STFTLoss(fft_sizes, hop_sizes, win_lengths,
                             loss_sc_weight=loss_sc_weight, loss_mag_weight=loss_mag_weight)

    # Create dummy audio data
    batch_size = 4
    sample_length = 22050 * 2 # 2 seconds
    y_true = torch.randn(batch_size, sample_length)
    y_pred = torch.randn(batch_size, sample_length) * 0.1 + y_true * 0.9 # Slightly noisy version

    # Calculate loss
    loss_sc, loss_mag, loss_total = stft_loss_fn(y_pred, y_true)

    print(f"Spectral Convergence Loss: {loss_sc.item():.4f}")
    print(f"Log STFT Magnitude Loss: {loss_mag.item():.4f}")
    print(f"Total STFT Loss: {loss_total.item():.4f}")