import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
import torchaudio.transforms as T
import io
from PIL import Image
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Import the enhanced UNet model
from models.unet_vocoder.model import UNetVocoder
from models.unet_vocoder.utils import generate_noise, audio_to_mel
from utils.plotting import plot_spectrograms_to_figure

# Signal Processing Enhancements
from torchaudio.functional import phase_vocoder
from torchaudio.transforms import MelScale, Spectrogram

# Helper functions for enhanced spectral loss
def spectral_convergence_loss(x_mag, y_mag):
    """Spectral convergence loss."""
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

def log_stft_magnitude_loss(x_mag, y_mag):
    """Log STFT magnitude loss."""
    # Add small epsilon to prevent log(0)
    eps = 1e-7
    return F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))

def phase_loss(x_phase, y_phase):
    """Phase consistency loss."""
    # Circular distance between phases
    phase_diff = torch.abs(torch.angle(torch.exp(1j * (x_phase - y_phase))))
    return torch.mean(phase_diff)

def apply_perceptual_weighting(spec_mag, sample_rate, curve_type='a', is_mel=False):
    """
    Apply perceptual weighting to spectral magnitude (either STFT or mel).
    
    Args:
        spec_mag: Magnitude spectrum [B, F, T] or mel spectrogram [B, M, T]
        sample_rate: Audio sample rate
        curve_type: Type of weighting curve ('a' for A-weighting)
        is_mel: Flag indicating if input is mel spectrogram
    
    Returns:
        Weighted spectrum with same shape as input
    """
    # Get frequency dimension directly from the input tensor
    freq_dim = spec_mag.shape[1]
    
    if is_mel:
        # For mel spectrograms, approximate mel band center frequencies
        # This is an approximation - ideally we'd use the actual mel filter centers
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (sample_rate/2) / 700)
        mel_points = torch.linspace(mel_min, mel_max, freq_dim, device=spec_mag.device)
        freqs = 700 * (10**(mel_points / 2595) - 1)
    else:
        # For STFT magnitudes, use linear frequency scale
        freqs = torch.linspace(0, sample_rate/2, freq_dim, device=spec_mag.device)
    
    # Define various weighting curves
    if curve_type == 'a':
        # A-weighting approximation (simplified)
        # Based on https://en.wikipedia.org/wiki/A-weighting
        f2 = torch.pow(freqs, 2)
        numerator = 12200**2 * f2**2
        denominator = (f2 + 20.6**2) * torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200**2)
        weights = 2.0 + 20 * torch.log10(numerator / denominator + 1e-8)
        # Normalize and convert to range [0, 1]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    else:
        # Default: equal weighting
        weights = torch.ones_like(freqs)
    
    # Apply weighting to each frequency bin
    weights = weights.view(1, -1, 1)  # [1, F, 1]
    weighted_spec = spec_mag * weights
    
    return weighted_spec

def extract_harmonics(audio, f0, sample_rate, harmonic_threshold=0.2):
    """
    Extract harmonic components using a comb filter based on F0.
    
    Args:
        audio: Time-domain audio signal [B, T, 1]
        f0: Fundamental frequency contour [B, T, 1]
        sample_rate: Audio sample rate
        harmonic_threshold: Threshold for harmonic band filters
        
    Returns:
        harmonic_part: Harmonic component [B, T, 1]
        noise_part: Noise residual [B, T, 1]
    """
    batch_size, time_len, _ = audio.shape
    
    # Flatten the audio for processing
    audio_flat = audio.squeeze(-1)  # [B, T]
    
    # Get STFT
    spec_transform = Spectrogram(
        n_fft=2048,
        hop_length=512,
        power=None,  # Return complex STFT
    )
    
    # Extract complex spectrum
    stft = spec_transform(audio_flat)  # [B, F, T_stft, 2] (real, imag)
    stft_complex = torch.complex(stft[..., 0], stft[..., 1])  # [B, F, T_stft]
    
    # Extract magnitudes and phases
    magnitudes = torch.abs(stft_complex)  # [B, F, T_stft]
    phases = torch.angle(stft_complex)  # [B, F, T_stft]
    
    # Prepare f0 to match STFT time resolution
    t_stft = stft_complex.shape[2]
    f0_downsampled = F.interpolate(f0.transpose(1, 2), size=t_stft, mode='linear', align_corners=False)
    f0_downsampled = f0_downsampled.transpose(1, 2)  # [B, T_stft, 1]
    
    # For each frame, create a harmonic mask based on f0
    harmonic_mask = torch.zeros_like(magnitudes)
    
    # Get frequency axis for bins
    n_fft = stft_complex.shape[1]
    freqs = torch.linspace(0, sample_rate/2, n_fft, device=audio.device)
    
    # For each batch and time step
    for b in range(batch_size):
        for t in range(t_stft):
            # Get the fundamental frequency for this frame
            f0_val = f0_downsampled[b, t, 0].item()
            
            if f0_val > 0:  # Only process for voiced frames
                # For each harmonic up to Nyquist frequency
                for h in range(1, int(sample_rate/2 / f0_val) + 1):
                    harmonic_freq = h * f0_val
                    
                    # Find frequency bins close to this harmonic
                    bin_dist = torch.abs(freqs - harmonic_freq)
                    
                    # Set mask to 1 for bins near harmonic
                    harmonic_width = harmonic_freq * harmonic_threshold
                    harmonic_mask[b, bin_dist < harmonic_width, t] = 1.0
    
    # Apply masks to get harmonic and noise components
    harmonic_spec = stft_complex * harmonic_mask
    noise_spec = stft_complex * (1 - harmonic_mask)
    
    # Convert back to time domain using inverse STFT
    harmonic_part = torch.istft(
        harmonic_spec,
        n_fft=2048,
        hop_length=512,
        length=time_len
    ).unsqueeze(-1)  # [B, T, 1]
    
    noise_part = torch.istft(
        noise_spec,
        n_fft=2048,
        hop_length=512,
        length=time_len
    ).unsqueeze(-1)  # [B, T, 1]
    
    return harmonic_part, noise_part


class VocoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Detach config from graph to avoid issues with saving hyperparameters
        self.config = config.copy()
        self.save_hyperparameters(self.config)
        
        # Initialize the enhanced vocoder model
        self.vocoder = UNetVocoder(self.config)
        self.use_f0 = self.config['vocoder'].get('use_f0_conditioning', False)
        
        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_fft = self.config['audio']['n_fft']
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio'].get('fmin', 0)
        self.fmax = self.config['audio'].get('fmax', self.sample_rate // 2)
        
        # Store noise scale for inference
        self.noise_scale = self.config['vocoder'].get('noise_scale', 0.6)
        
        # Signal Processing Enhancements
        # 1. Phase reconstruction
        self.use_phase_prediction = self.config['vocoder'].get('use_phase_prediction', False)
        self.phase_loss_weight = self.config['vocoder'].get('phase_loss_weight', 0.5)
        
        # 2. Perceptual weighting
        self.use_perceptual_weighting = self.config['vocoder'].get('use_perceptual_weighting', False)
        self.perceptual_curve_type = self.config['vocoder'].get('perceptual_curve_type', 'a')
        
        # 3. Harmonic-plus-noise model
        self.use_harmonic_plus_noise = self.config['vocoder'].get('use_harmonic_plus_noise', False)
        self.harmonic_ratio = self.config['vocoder'].get('harmonic_ratio', 0.7)
        self.harmonic_loss_weight = self.config['vocoder'].get('harmonic_loss_weight', 0.6)
        self.noise_loss_weight = self.config['vocoder'].get('noise_loss_weight', 0.4)
        self.harmonic_threshold = self.config['vocoder'].get('harmonic_threshold', 0.2)
        
        # Loss configuration
        self.loss_type = self.config['train'].get('vocoder_loss', 'Combined')
        
        # Loss weights
        self.lambda_td = self.config['train'].get('loss_lambda_td', 1.0)
        self.lambda_sc = self.config['train'].get('loss_lambda_sc', 0.0)
        self.lambda_mag = self.config['train'].get('loss_lambda_mag', 0.0)
        self.lambda_mel = self.config['train'].get('loss_lambda_mel', 0.0)

        # STFT parameters for loss calculation
        self.stft_fft_size = self.config['train'].get('stft_fft_size', self.n_fft)
        self.stft_hop_size = self.config['train'].get('stft_hop_size', self.hop_length)
        self.stft_win_length = self.config['train'].get('stft_win_length', self.win_length)
        
        # Create STFT operator
        self.stft = T.Spectrogram(
            n_fft=self.stft_fft_size,
            hop_length=self.stft_hop_size,
            win_length=self.stft_win_length,
            window_fn=torch.hann_window,
            power=None,  # Return complex result
            center=True,
            pad_mode="reflect",
            normalized=False,
        )
        
        # Create Mel Spectrogram operator for reconstruction loss
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.fmin,
            f_max=self.fmax,
            n_mels=self.n_mels,
            power=1.0,
            center=True,
            pad_mode="reflect",
            norm='slaney',
            mel_scale="slaney"
        )
        
        # Multi-resolution STFT parameters
        self.use_multi_resolution_stft = self.config['vocoder'].get('use_multi_resolution_stft', False)
        if self.use_multi_resolution_stft:
            # Define multiple STFT resolutions
            self.stft_resolutions = [
                {"n_fft": 512, "hop_length": 128, "win_length": 512},
                {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
                {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
            ]
            # Create STFT operators for each resolution
            self.stft_transforms = nn.ModuleList([
                T.Spectrogram(
                    n_fft=params["n_fft"],
                    hop_length=params["hop_length"],
                    win_length=params["win_length"],
                    window_fn=torch.hann_window,
                    power=None,
                    center=True,
                    pad_mode="reflect",
                    normalized=False,
                ) for params in self.stft_resolutions
            ])
    
    def forward(self, mel_spec, f0=None):
        """
        Forward pass through the vocoder model with autoregressive generation.

        Args:
            mel_spec (torch.Tensor): Mel spectrogram [B, T_mel, M]
            f0 (torch.Tensor, optional): Aligned F0 contour [B, T_mel, 1]. Required if use_f0 is True.

        Returns:
            torch.Tensor: Generated waveform [B, T_audio, 1] where T_audio is approx T_mel * hop_length
        """
        batch_size, time_steps, _ = mel_spec.size()
        
        # Generate noise with the SAME time dimension as mel_spec
        noise = torch.randn(batch_size, time_steps, 1, device=mel_spec.device) * self.noise_scale
        
        # Check if we're using autoregressive generation
        use_autoregressive = self.config['vocoder'].get('use_autoregressive', False)
        
        if use_autoregressive:
            # For autoregressive generation, we need to process the input chunk by chunk
            # Get chunk size from config (default to a reasonable size)
            chunk_size = self.config['vocoder'].get('autoregressive_chunk_size', 32)
            overlap = self.config['vocoder'].get('autoregressive_overlap', 8)
            
            # Initialize state
            state = None
            
            # Initialize output waveform
            total_frames = time_steps
            total_samples = total_frames * self.hop_length
            output_waveform = torch.zeros(batch_size, total_samples, 1, device=mel_spec.device)
            
            # Process in chunks with overlap
            for start_idx in range(0, total_frames, chunk_size - overlap):
                end_idx = min(start_idx + chunk_size, total_frames)
                
                # Get current chunk
                mel_chunk = mel_spec[:, start_idx:end_idx, :]
                noise_chunk = noise[:, start_idx:end_idx, :]
                
                # Get f0 chunk if available
                f0_chunk = None
                if self.use_f0 and f0 is not None:
                    f0_chunk = f0[:, start_idx:end_idx, :]
                
                # Generate waveform and update state
                chunk_output, state = self.vocoder(mel_chunk, noise_chunk, f0=f0_chunk, state=state)
                
                # Calculate output indices for this chunk
                out_start = start_idx * self.hop_length
                out_end = end_idx * self.hop_length
                
                # If not the first chunk, apply cross-fade for overlap
                if start_idx > 0:
                    # Calculate overlap region
                    overlap_samples = overlap * self.hop_length
                    fade_in = torch.linspace(0, 1, overlap_samples, device=mel_spec.device)
                    fade_out = 1 - fade_in
                    
                    # Get overlap region indices
                    overlap_start = out_start
                    overlap_end = overlap_start + overlap_samples
                    
                    # Apply cross-fade
                    fade_in = fade_in.view(1, -1, 1)
                    fade_out = fade_out.view(1, -1, 1)
                    
                    output_waveform[:, overlap_start:overlap_end, :] = (
                        output_waveform[:, overlap_start:overlap_end, :] * fade_out +
                        chunk_output[:, :overlap_samples, :] * fade_in
                    )
                    
                    # Copy non-overlap region
                    output_waveform[:, overlap_end:out_end, :] = chunk_output[:, overlap_samples:, :]
                else:
                    # First chunk, just copy
                    output_waveform[:, out_start:out_end, :] = chunk_output
            
            return output_waveform
        
        else:
            # Non-autoregressive mode, use normal forward pass
            # Generate waveform with the UNetVocoder
            if self.use_f0:
                if f0 is None:
                    raise ValueError("F0 conditioning is enabled, but f0 tensor was not provided.")
                waveform = self.vocoder(mel_spec, noise, f0=f0)
            else:
                waveform = self.vocoder(mel_spec, noise)
                
            return waveform
    
    def _process_batch(self, batch):
        """
        Process batch data to extract relevant inputs and targets
        """
        mel_spec = batch.get('mel_spec')
        target_audio = batch.get('target_audio')
        lengths = batch.get('length')  # Original audio lengths before padding
        f0 = batch.get('f0', None)  # Get F0 using key from config
        
        # Reshape target audio if needed: [B, T_audio] -> [B, T_audio, 1]
        if target_audio is not None and target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(-1)
            
        # Reshape f0 if needed: [B, T_mel] -> [B, T_mel, 1]
        if f0 is not None and f0.dim() == 2:
            f0 = f0.unsqueeze(-1)

        # Ensure f0 is on the same device as mel_spec
        if f0 is not None and mel_spec is not None and f0.device != mel_spec.device:
             f0 = f0.to(mel_spec.device)
             
        return mel_spec, target_audio, f0, lengths
    
    def _ensure_same_mel_length(self, pred, target):
        """ Adjusts pred tensor length (dim 2) to match target tensor length (dim 2) for Mel Spectrograms.
            Pads or truncates pred; target remains unchanged.
        """
        pred_len = pred.shape[2] # Use time dimension (index 2)
        target_len = target.shape[2] # Use time dimension (index 2)

        if pred_len == target_len:
            return pred, target

        if self.training and hasattr(self, 'global_step') and self.global_step % 500 == 0: # Log less frequently
             print(f"Warning: Mel length mismatch. Adjusting Pred: {pred_len} to Target: {target_len}.")

        if pred_len > target_len:
            # Truncate prediction
            pred = pred[:, :, :target_len]
        else: # pred_len < target_len
            # Pad prediction
            padding = target_len - pred_len
            # Pad the end of the last dimension (time)
            pad_dims = [0] * (pred.dim() * 2) # e.g., [0, 0, 0, 0, 0, 0] for 3D
            pad_dims[1] = padding # Pad end of last dim
            pred = F.pad(pred, pad_dims)

        return pred, target # Return modified pred and original target

    def _ensure_same_time_domain_length(self, pred, target):
        """ Adjusts pred tensor length to match target tensor length along the time dimension (last dim).
            Pads or truncates pred; target remains unchanged. Assumes input is [B, T] or [T].
        """
        pred_len = pred.shape[-1] # Use last dimension (time)
        target_len = target.shape[-1] # Use last dimension (time)

        if pred_len == target_len:
            return pred, target

        if self.training and hasattr(self, 'global_step') and self.global_step % 500 == 0: # Log less frequently
             print(f"Warning: Time domain length mismatch. Adjusting Pred: {pred_len} to Target: {target_len}.")

        if pred_len > target_len:
            # Truncate prediction
            pred = pred[..., :target_len] # Use ellipsis for flexibility with dims
        else: # pred_len < target_len
            # Pad prediction
            padding = target_len - pred_len
            # Pad the end of the last dimension
            pad_dims = [0] * (pred.dim() * 2)
            pad_dims[0] = 0 # No padding at the start of the last dim
            pad_dims[1] = padding # Pad end of last dim
            pred = F.pad(pred, pad_dims)

        return pred, target # Return modified pred and original target


    def time_domain_loss(self, y_pred, y_true):
        """ Compute L1 time-domain loss. Handles length mismatch by trimming. """
        y_pred_adj, y_true_adj = self._ensure_same_time_domain_length(y_pred, y_true)
        return F.l1_loss(y_pred_adj, y_true_adj)

    def stft_loss(self, y_pred, y_true, apply_weighting=False):
        """ Compute Single-Resolution STFT loss (SC + Mag). """
        # Ensure inputs are flat [B, T_audio]
        y_pred_flat = y_pred.squeeze(-1)
        y_true_flat = y_true.squeeze(-1)
        
        # Trim to same length before STFT
        y_pred_flat, y_true_flat = self._ensure_same_time_domain_length(y_pred_flat, y_true_flat)

        # Calculate STFT
        stft_pred = self.stft(y_pred_flat)
        stft_true = self.stft(y_true_flat)
        
        # Check if the result is a complex tensor or a real tensor with last dim of size 2
        if torch.is_complex(stft_pred):
            # Handle complex tensor
            stft_pred_mag = torch.abs(stft_pred)
            stft_true_mag = torch.abs(stft_true)
            
            # Get phases if phase prediction is enabled
            if self.use_phase_prediction:
                stft_pred_phase = torch.angle(stft_pred)
                stft_true_phase = torch.angle(stft_true)
        else:
            # Handle real tensor with last dim of size 2 (real, imag)
            stft_pred_mag = torch.sqrt(stft_pred.pow(2).sum(-1) + 1e-9)
            stft_true_mag = torch.sqrt(stft_true.pow(2).sum(-1) + 1e-9)
            
            # Get phases if phase prediction is enabled
            if self.use_phase_prediction:
                stft_pred_phase = torch.atan2(stft_pred[..., 1], stft_pred[..., 0])
                stft_true_phase = torch.atan2(stft_true[..., 1], stft_true[..., 0])
        
        # Apply perceptual weighting if enabled
        if apply_weighting and self.use_perceptual_weighting:
            stft_pred_mag = apply_perceptual_weighting(
                stft_pred_mag, self.sample_rate, self.perceptual_curve_type)
            stft_true_mag = apply_perceptual_weighting(
                stft_true_mag, self.sample_rate, self.perceptual_curve_type)

        # Calculate phase loss
        if self.use_phase_prediction:
            phase_loss_val = phase_loss(stft_pred_phase, stft_true_phase)
        else:
            phase_loss_val = torch.tensor(0.0, device=y_pred.device)

        # Calculate loss components
        sc_loss = spectral_convergence_loss(stft_pred_mag, stft_true_mag)
        mag_loss = log_stft_magnitude_loss(stft_pred_mag, stft_true_mag)
        
        return sc_loss, mag_loss, phase_loss_val
    
    def multi_resolution_stft_loss(self, y_pred, y_true, apply_weighting=False):
        """Compute Multi-Resolution STFT loss across different FFT sizes."""
        # Ensure inputs are flat [B, T_audio]
        y_pred_flat = y_pred.squeeze(-1)
        y_true_flat = y_true.squeeze(-1)
        
        # Trim to same length before STFT
        y_pred_flat, y_true_flat = self._ensure_same_time_domain_length(y_pred_flat, y_true_flat)
        
        sc_losses = []
        mag_losses = []
        phase_losses = []
        
        # Calculate STFT for each resolution
        for stft_transform in self.stft_transforms:
            stft_pred = stft_transform(y_pred_flat)
            stft_true = stft_transform(y_true_flat)
            
            # Check if the result is a complex tensor or a real tensor with last dim of size 2
            if torch.is_complex(stft_pred):
                # Handle complex tensor
                stft_pred_mag = torch.abs(stft_pred)
                stft_true_mag = torch.abs(stft_true)
                
                # Get phases if phase prediction is enabled
                if self.use_phase_prediction:
                    stft_pred_phase = torch.angle(stft_pred)
                    stft_true_phase = torch.angle(stft_true)
            else:
                # Handle real tensor with last dim of size 2 (real, imag)
                stft_pred_mag = torch.sqrt(stft_pred.pow(2).sum(-1) + 1e-9)
                stft_true_mag = torch.sqrt(stft_true.pow(2).sum(-1) + 1e-9)
                
                # Get phases if phase prediction is enabled
                if self.use_phase_prediction:
                    stft_pred_phase = torch.atan2(stft_pred[..., 1], stft_pred[..., 0])
                    stft_true_phase = torch.atan2(stft_true[..., 1], stft_true[..., 0])
            
            # Apply perceptual weighting if enabled
            if apply_weighting and self.use_perceptual_weighting:
                stft_pred_mag = apply_perceptual_weighting(
                    stft_pred_mag, self.sample_rate, self.perceptual_curve_type, False)
                stft_true_mag = apply_perceptual_weighting(
                    stft_true_mag, self.sample_rate, self.perceptual_curve_type, False)
            
            # Calculate losses
            sc_losses.append(spectral_convergence_loss(stft_pred_mag, stft_true_mag))
            mag_losses.append(log_stft_magnitude_loss(stft_pred_mag, stft_true_mag))
            
            # Calculate phase loss if enabled
            if self.use_phase_prediction:
                phase_losses.append(phase_loss(stft_pred_phase, stft_true_phase))
        
        # Average losses across resolutions
        sc_loss = torch.mean(torch.stack(sc_losses))
        mag_loss = torch.mean(torch.stack(mag_losses))
        
        if self.use_phase_prediction:
            phase_loss_val = torch.mean(torch.stack(phase_losses))
        else:
            phase_loss_val = torch.tensor(0.0, device=y_pred.device)
            
        return sc_loss, mag_loss, phase_loss_val

    def mel_reconstruction_loss(self, generated_audio, target_mel):
        """ Compute Mel Spectrogram Reconstruction Loss. """
        # Ensure generated_audio is flat [B, T_audio]
        gen_audio_flat = generated_audio.squeeze(-1)
        
        # Generate mel from audio
        gen_mel = self.mel_spectrogram_transform(gen_audio_flat) # [B, M, T_mel_gen]
        
        # Target mel is likely [B, T_mel_target, M], transpose it
        target_mel_transposed = target_mel.transpose(1, 2) # [B, M, T_mel_target]
        
        # Ensure same time dimension
        gen_mel_adj, target_mel_adj = self._ensure_same_mel_length(gen_mel, target_mel_transposed)
        
        # Apply perceptual weighting if enabled
        if self.use_perceptual_weighting:
            gen_mel_adj = apply_perceptual_weighting(
                gen_mel_adj, self.sample_rate, self.perceptual_curve_type, is_mel=True)
            target_mel_adj = apply_perceptual_weighting(
                target_mel_adj, self.sample_rate, self.perceptual_curve_type, is_mel=True)
        
        # Calculate L1 loss
        mel_loss = F.l1_loss(gen_mel_adj, target_mel_adj)
        return mel_loss
    
    def harmonic_plus_noise_loss(self, generated_audio, target_audio, f0):
        """
        Compute harmonic and noise-specific losses using harmonic-plus-noise decomposition.
        """
        # Extract harmonic and noise components
        gen_harmonic, gen_noise = extract_harmonics(
            generated_audio, f0, self.sample_rate, self.harmonic_threshold)
        
        target_harmonic, target_noise = extract_harmonics(
            target_audio, f0, self.sample_rate, self.harmonic_threshold)
        
        # Compute L1 losses for each component
        harmonic_loss = F.l1_loss(gen_harmonic, target_harmonic)
        noise_loss = F.l1_loss(gen_noise, target_noise)
        
        # Weight and combine
        combined_loss = (self.harmonic_loss_weight * harmonic_loss + 
                         self.noise_loss_weight * noise_loss)
        
        return combined_loss, harmonic_loss, noise_loss
    
    def _calculate_combined_loss(self, generated_audio, target_audio, mel_spec, f0=None):
        """ Calculates all loss components and combines them. """
        loss = 0
        losses = {}

        # 1. Time Domain Loss (L1)
        if self.lambda_td > 0:
            td_loss = self.time_domain_loss(generated_audio, target_audio)
            loss += self.lambda_td * td_loss
            losses['td_loss'] = td_loss
        else:
             losses['td_loss'] = torch.tensor(0.0, device=generated_audio.device)

        # 2. STFT Loss (SC + Mag + Phase)
        if self.lambda_sc > 0 or self.lambda_mag > 0:
            if self.use_multi_resolution_stft:
                sc_loss, mag_loss, phase_loss_val = self.multi_resolution_stft_loss(
                    generated_audio, target_audio, self.use_perceptual_weighting)
            else:
                sc_loss, mag_loss, phase_loss_val = self.stft_loss(
                    generated_audio, target_audio, self.use_perceptual_weighting)
            
            if self.lambda_sc > 0:
                loss += self.lambda_sc * sc_loss
                losses['sc_loss'] = sc_loss
            else:
                 losses['sc_loss'] = torch.tensor(0.0, device=generated_audio.device)
                 
            if self.lambda_mag > 0:
                loss += self.lambda_mag * mag_loss
                losses['mag_loss'] = mag_loss
            else:
                 losses['mag_loss'] = torch.tensor(0.0, device=generated_audio.device)
                 
            if self.use_phase_prediction and self.phase_loss_weight > 0:
                loss += self.phase_loss_weight * phase_loss_val
                losses['phase_loss'] = phase_loss_val
            else:
                losses['phase_loss'] = torch.tensor(0.0, device=generated_audio.device)
        else:
             losses['sc_loss'] = torch.tensor(0.0, device=generated_audio.device)
             losses['mag_loss'] = torch.tensor(0.0, device=generated_audio.device)
             losses['phase_loss'] = torch.tensor(0.0, device=generated_audio.device)

        # 3. Mel Reconstruction Loss
        if self.lambda_mel > 0:
            mel_loss = self.mel_reconstruction_loss(generated_audio, mel_spec)
            loss += self.lambda_mel * mel_loss
            losses['mel_loss'] = mel_loss
        else:
             losses['mel_loss'] = torch.tensor(0.0, device=generated_audio.device)
             
        # 4. Harmonic-Plus-Noise Loss (if enabled and f0 is available)
        if self.use_harmonic_plus_noise and f0 is not None:
            hn_loss, h_loss, n_loss = self.harmonic_plus_noise_loss(
                generated_audio, target_audio, f0)
            loss += hn_loss
            losses['harmonic_loss'] = h_loss
            losses['noise_loss'] = n_loss
            losses['hn_combined_loss'] = hn_loss
        else:
            losses['harmonic_loss'] = torch.tensor(0.0, device=generated_audio.device)
            losses['noise_loss'] = torch.tensor(0.0, device=generated_audio.device)
            losses['hn_combined_loss'] = torch.tensor(0.0, device=generated_audio.device)
             
        losses['total_loss'] = loss
        return losses

    def training_step(self, batch, batch_idx):
        mel_spec, target_audio, f0, lengths = self._process_batch(batch)
        
        # Forward pass
        generated_audio = self(mel_spec, f0=f0 if self.use_f0 else None)
        
        # Calculate losses
        losses = self._calculate_combined_loss(generated_audio, target_audio, mel_spec, f0)
        
        # Log losses
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/td_loss', losses['td_loss'], on_step=True, on_epoch=True)
        self.log('train/sc_loss', losses['sc_loss'], on_step=True, on_epoch=True)
        self.log('train/mag_loss', losses['mag_loss'], on_step=True, on_epoch=True)
        self.log('train/mel_loss', losses['mel_loss'], on_step=True, on_epoch=True)
        
        if self.use_phase_prediction:
            self.log('train/phase_loss', losses['phase_loss'], on_step=True, on_epoch=True)
            
        if self.use_harmonic_plus_noise:
            self.log('train/harmonic_loss', losses['harmonic_loss'], on_step=True, on_epoch=True)
            self.log('train/noise_loss', losses['noise_loss'], on_step=True, on_epoch=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        mel_spec, target_audio, f0, lengths = self._process_batch(batch)
        
        # Forward pass
        generated_audio = self(mel_spec, f0=f0 if self.use_f0 else None)
        
        # Calculate losses
        losses = self._calculate_combined_loss(generated_audio, target_audio, mel_spec, f0)
        
        # Log losses
        self.log('val/loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/td_loss', losses['td_loss'], on_step=False, on_epoch=True)
        self.log('val/sc_loss', losses['sc_loss'], on_step=False, on_epoch=True)
        self.log('val/mag_loss', losses['mag_loss'], on_step=False, on_epoch=True)
        self.log('val/mel_loss', losses['mel_loss'], on_step=False, on_epoch=True)
        
        if self.use_phase_prediction:
            self.log('val/phase_loss', losses['phase_loss'], on_step=False, on_epoch=True)
            
        if self.use_harmonic_plus_noise:
            self.log('val/harmonic_loss', losses['harmonic_loss'], on_step=False, on_epoch=True)
            self.log('val/noise_loss', losses['noise_loss'], on_step=False, on_epoch=True)
        
        # Log audio samples every N epochs
        if batch_idx == 0 and self.current_epoch % self.config['train'].get('log_vocoder_audio_epoch_interval', 5) == 0:
            self._log_audio_samples(mel_spec, generated_audio, target_audio)
            self._log_mel_comparison(mel_spec, generated_audio)
            
            # If harmonic-plus-noise model is enabled, visualize components
            if self.use_harmonic_plus_noise and f0 is not None:
                self._log_harmonic_noise_components(generated_audio, target_audio, f0)
            
        return losses
    
    def _log_audio_samples(self, mel_spec, generated_audio, target_audio):
        """Log audio samples and spectrograms for visualization"""
        # Only log the first few samples in the batch
        num_samples = min(2, mel_spec.size(0))
        
        for idx in range(num_samples):
            # Get original mel spectrogram for visualization
            mel_to_plot = mel_spec[idx].detach().cpu().numpy().T  # [M, T]
            
            # Plot the mel spectrogram
            fig = plt.figure(figsize=(10, 4))
            plt.imshow(mel_to_plot, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(f"Input Mel Spectrogram - Sample {idx}")
            plt.tight_layout()
            
            # Convert to image and log to tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            # Convert to tensor and log
            img = Image.open(buf)
            img_tensor = torchvision.transforms.ToTensor()(img)
            
            # Log to tensorboard
            if self.logger:
                self.logger.experiment.add_image(
                    f'mel_spec/sample_{idx}',
                    img_tensor,
                    self.global_step
                )
                
                # Log audio samples
                if hasattr(self.logger.experiment, 'add_audio'):
                    # Get audio samples
                    gen_audio = generated_audio[idx, :, 0].detach().cpu()
                    tgt_audio = target_audio[idx, :, 0].detach().cpu()
                    
                    # Normalize for audio logging
                    gen_audio = gen_audio / (gen_audio.abs().max() + 1e-6)
                    tgt_audio = tgt_audio / (tgt_audio.abs().max() + 1e-6)
                    
                    # Ensure both have same length for comparison
                    min_len = min(gen_audio.size(0), tgt_audio.size(0))
                    gen_audio = gen_audio[:min_len]
                    tgt_audio = tgt_audio[:min_len]
                    
                    # Log audio at full sample rate
                    self.logger.experiment.add_audio(
                        f'audio/generated_{idx}',
                        gen_audio,
                        self.global_step,
                        sample_rate=self.sample_rate
                    )
                    self.logger.experiment.add_audio(
                        f'audio/target_{idx}',
                        tgt_audio,
                        self.global_step,
                        sample_rate=self.sample_rate
                    )
    
    def _log_mel_comparison(self, input_mel, generated_audio):
        """Create and log a visualization comparing input mel spectrograms with 
        mel spectrograms generated from the predicted audio"""
        # Only use the first sample for visualization
        idx = 0
        
        # Get the input mel spectrogram
        input_mel_np = input_mel[idx].detach().cpu().numpy().T  # [M, T]
        
        # Convert generated audio back to mel spectrogram
        gen_audio_flat = generated_audio[idx, :, 0].detach().cpu().unsqueeze(0)  # [1, T*hop_length]
        
        # Use the initialized transform for consistency
        with torch.no_grad():
             gen_mel = self.mel_spectrogram_transform(gen_audio_flat.to(self.device))
             gen_mel_np = gen_mel[0].cpu().numpy()  # [M, T_mel_gen]
        
        # Get shapes for comparison
        input_shape = input_mel_np.shape
        gen_shape = gen_mel_np.shape
        
        # If lengths don't match, resize for visualization
        if gen_shape[1] != input_shape[1]:
            from scipy.ndimage import zoom
            scale_factor = input_shape[1] / gen_shape[1]
            gen_mel_np = zoom(gen_mel_np, (1, scale_factor), order=1)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot input mel spectrogram on top
        im0 = axes[0].imshow(input_mel_np, aspect='auto', origin='lower')
        axes[0].set_title("Input Mel Spectrogram")
        axes[0].set_ylabel("Mel Bins")
        fig.colorbar(im0, ax=axes[0])
        
        # Plot generated mel spectrogram on bottom
        im1 = axes[1].imshow(gen_mel_np, aspect='auto', origin='lower')
        axes[1].set_title("Generated Mel Spectrogram")
        axes[1].set_ylabel("Mel Bins")
        axes[1].set_xlabel("Frames")
        fig.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        
        # Convert to image and log to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Convert to tensor and log
        img = Image.open(buf)
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # Log to tensorboard
        if self.logger:
            self.logger.experiment.add_image(
                f'mel_comparison/sample_{idx}',
                img_tensor,
                self.global_step
            )
    
    def _log_harmonic_noise_components(self, generated_audio, target_audio, f0):
        """Visualize and log harmonic and noise components"""
        # Only use the first sample
        idx = 0
        
        # Get components
        with torch.no_grad():
            gen_harmonic, gen_noise = extract_harmonics(
                generated_audio[idx:idx+1], 
                f0[idx:idx+1], 
                self.sample_rate,
                self.harmonic_threshold
            )
            
            target_harmonic, target_noise = extract_harmonics(
                target_audio[idx:idx+1],
                f0[idx:idx+1],
                self.sample_rate,
                self.harmonic_threshold
            )
        
        # Compute spectrograms for visualization
        stft_transform = Spectrogram(
            n_fft=1024,
            hop_length=256,
            power=2.0  # Power spectrogram
        )
        
        # Convert to CPU and calculate spectrograms
        gen_h_spec = stft_transform(gen_harmonic.squeeze(-1).cpu())[0]
        gen_n_spec = stft_transform(gen_noise.squeeze(-1).cpu())[0]
        tgt_h_spec = stft_transform(target_harmonic.squeeze(-1).cpu())[0]
        tgt_n_spec = stft_transform(target_noise.squeeze(-1).cpu())[0]
        
        # Convert to dB scale for better visualization
        def to_db(spec):
            return 10 * torch.log10(spec + 1e-9)
        
        gen_h_spec_db = to_db(gen_h_spec).numpy()
        gen_n_spec_db = to_db(gen_n_spec).numpy()
        tgt_h_spec_db = to_db(tgt_h_spec).numpy()
        tgt_n_spec_db = to_db(tgt_n_spec).numpy()
        
        # Create figure with four subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot spectrograms
        im0 = axes[0, 0].imshow(tgt_h_spec_db, aspect='auto', origin='lower')
        axes[0, 0].set_title("Target Harmonic Component")
        fig.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(gen_h_spec_db, aspect='auto', origin='lower')
        axes[0, 1].set_title("Generated Harmonic Component")
        fig.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[1, 0].imshow(tgt_n_spec_db, aspect='auto', origin='lower')
        axes[1, 0].set_title("Target Noise Component")
        fig.colorbar(im2, ax=axes[1, 0])
        
        im3 = axes[1, 1].imshow(gen_n_spec_db, aspect='auto', origin='lower')
        axes[1, 1].set_title("Generated Noise Component")
        fig.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Convert to image and log to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Convert to tensor and log
        img = Image.open(buf)
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # Log to tensorboard
        if self.logger:
            self.logger.experiment.add_image(
                f'harmonic_noise_decomposition/sample_{idx}',
                img_tensor,
                self.global_step
            )
            
            # Log audio samples of components
            if hasattr(self.logger.experiment, 'add_audio'):
                # Normalize and prepare audio samples
                components = {
                    'harmonic/target': target_harmonic[0, :, 0].detach().cpu(),
                    'harmonic/generated': gen_harmonic[0, :, 0].detach().cpu(),
                    'noise/target': target_noise[0, :, 0].detach().cpu(),
                    'noise/generated': gen_noise[0, :, 0].detach().cpu()
                }
                
                # Log each component
                for name, audio in components.items():
                    # Normalize
                    audio = audio / (audio.abs().max() + 1e-6)
                    # Log
                    self.logger.experiment.add_audio(
                        f'components/{name}',
                        audio,
                        self.global_step,
                        sample_rate=self.sample_rate
                    )
    
    def configure_optimizers(self):
        # Get learning rate from config
        lr = self.config['train'].get('vocoder_learning_rate', 0.0002)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=(0.9, 0.999)
        )
        
        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }