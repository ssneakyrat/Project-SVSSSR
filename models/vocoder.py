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

from models.unet_vocoder.model import UNetVocoder
from models.unet_vocoder.utils import generate_noise, audio_to_mel # Assuming audio_to_mel is suitable
from utils.plotting import plot_spectrograms_to_figure

# Helper function for STFT Loss
def spectral_convergence_loss(x_mag, y_mag):
    """Spectral convergence loss."""
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

def log_stft_magnitude_loss(x_mag, y_mag):
    """Log STFT magnitude loss."""
    # Add small epsilon to prevent log(0)
    eps = 1e-7
    return F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))

class VocoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Detach config from graph to avoid issues with saving hyperparameters
        self.config = config.copy()
        self.save_hyperparameters(self.config)
        
        # Initialize the vocoder model
        self.vocoder = UNetVocoder(self.config)
        self.use_f0 = self.config['vocoder'].get('use_f0_conditioning', False)
        
        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_fft = self.config['audio']['n_fft'] # Use n_fft from config
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio'].get('fmin', 0)
        self.fmax = self.config['audio'].get('fmax', self.sample_rate // 2)
        
        # Store noise scale for inference
        self.noise_scale = self.config['vocoder'].get('noise_scale', 0.6)
        
        # Loss configuration
        self.loss_type = self.config['train'].get('vocoder_loss', 'Combined') # Default to combined
        
        # Loss weights
        self.lambda_td = self.config['train'].get('loss_lambda_td', 1.0)
        self.lambda_sc = self.config['train'].get('loss_lambda_sc', 0.0) # Default to 0 if not specified
        self.lambda_mag = self.config['train'].get('loss_lambda_mag', 0.0) # Default to 0 if not specified
        self.lambda_mel = self.config['train'].get('loss_lambda_mel', 0.0) # Default to 0 if not specified

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
            power=None, # Return complex result
            center=True,
            pad_mode="reflect",
            normalized=False,
        )
        
        # Create Mel Spectrogram operator for reconstruction loss
        # Ensure parameters match those used for input mel generation
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.fmin,
            f_max=self.fmax,
            n_mels=self.n_mels,
            power=1.0, # Use power=1 for magnitude mels if input mels are magnitude based
            center=True,
            pad_mode="reflect",
            norm='slaney', # Common normalization
            mel_scale="slaney" # Common mel scale
        )
    def forward(self, mel_spec, f0=None):
        """
        Forward pass through the vocoder model.

        Args:
            mel_spec (torch.Tensor): Mel spectrogram [B, T_mel, M]
            f0 (torch.Tensor, optional): Aligned F0 contour [B, T_mel, 1]. Required if use_f0 is True.

        Returns:
            torch.Tensor: Generated waveform [B, T_audio, 1] where T_audio is approx T_mel * hop_length
        """
        batch_size, time_steps, _ = mel_spec.size()
        
        # Generate noise with the SAME time dimension as mel_spec
        noise = torch.randn(batch_size, time_steps, 1, device=mel_spec.device) * self.noise_scale
        
        # Generate waveform with the UNetVocoder
        # The UNetVocoder now handles the internal upsampling to audio rate
        # Pass f0 if conditioning is enabled
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
        f0 = batch.get(self.config['data'].get('f0_key', 'f0_contour'), None) # Get F0 using key from config
        
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
            # F.pad expects pads for (last_dim_start, last_dim_end, second_last_dim_start, ...)
            pad_dims = [0] * (pred.dim() * 2)
            pad_dims[0] = 0 # No padding at the start of the last dim
            pad_dims[1] = padding # Pad end of last dim
            pred = F.pad(pred, pad_dims)

        return pred, target # Return modified pred and original target


    def time_domain_loss(self, y_pred, y_true):
        """ Compute L1 time-domain loss. Handles length mismatch by trimming. """
        y_pred_adj, y_true_adj = self._ensure_same_time_domain_length(y_pred, y_true)
        return F.l1_loss(y_pred_adj, y_true_adj)

    def stft_loss(self, y_pred, y_true):
        """ Compute Single-Resolution STFT loss (SC + Mag). """
        # Ensure inputs are flat [B, T_audio]
        y_pred_flat = y_pred.squeeze(-1)
        y_true_flat = y_true.squeeze(-1)
        
        # Trim to same length before STFT
        y_pred_flat, y_true_flat = self._ensure_same_time_domain_length(y_pred_flat, y_true_flat)

        # Calculate STFT
        stft_pred = self.stft(y_pred_flat) # [B, F, T_stft, 2] for complex
        stft_true = self.stft(y_true_flat) # [B, F, T_stft, 2] for complex
        
        # Get magnitudes
        stft_pred_mag = torch.sqrt(stft_pred.pow(2).sum(-1) + 1e-9) # [B, F, T_stft]
        stft_true_mag = torch.sqrt(stft_true.pow(2).sum(-1) + 1e-9) # [B, F, T_stft]

        # Calculate loss components
        sc_loss = spectral_convergence_loss(stft_pred_mag, stft_true_mag)
        mag_loss = log_stft_magnitude_loss(stft_pred_mag, stft_true_mag)
        
        return sc_loss, mag_loss

    def mel_reconstruction_loss(self, generated_audio, target_mel):
        """ Compute Mel Spectrogram Reconstruction Loss. """
        # Ensure generated_audio is flat [B, T_audio]
        gen_audio_flat = generated_audio.squeeze(-1)
        
        # Generate mel from audio
        # Need to handle potential length differences between input mel and mel from generated audio
        # Option 1: Pad/trim audio before mel transform (might be inaccurate if lengths differ significantly)
        # Option 2: Pad/trim mel spectrograms after transform (preferred)
        
        gen_mel = self.mel_spectrogram_transform(gen_audio_flat) # [B, M, T_mel_gen]
        
        # Target mel is likely [B, T_mel_target, M], transpose it
        target_mel_transposed = target_mel.transpose(1, 2) # [B, M, T_mel_target]
        
        # Ensure same time dimension (T_mel) by padding/trimming the shorter one
        gen_mel_adj, target_mel_adj = self._ensure_same_mel_length(gen_mel, target_mel_transposed) # Adjusts gen_mel (dim 2) to match target_mel (dim 2) length
        
        # Calculate L1 loss
        mel_loss = F.l1_loss(gen_mel_adj, target_mel_adj)
        return mel_loss
    
    def _calculate_combined_loss(self, generated_audio, target_audio, mel_spec):
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

        # 2. STFT Loss (SC + Mag)
        if self.lambda_sc > 0 or self.lambda_mag > 0:
            sc_loss, mag_loss = self.stft_loss(generated_audio, target_audio)
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
        else:
             losses['sc_loss'] = torch.tensor(0.0, device=generated_audio.device)
             losses['mag_loss'] = torch.tensor(0.0, device=generated_audio.device)

        # 3. Mel Reconstruction Loss
        if self.lambda_mel > 0:
            mel_loss = self.mel_reconstruction_loss(generated_audio, mel_spec)
            loss += self.lambda_mel * mel_loss
            losses['mel_loss'] = mel_loss
        else:
             losses['mel_loss'] = torch.tensor(0.0, device=generated_audio.device)
             
        losses['total_loss'] = loss
        return losses

    def training_step(self, batch, batch_idx):
        mel_spec, target_audio, f0, lengths = self._process_batch(batch)
        
        # Forward pass
        generated_audio = self(mel_spec, f0=f0 if self.use_f0 else None)
        
        # Calculate losses
        losses = self._calculate_combined_loss(generated_audio, target_audio, mel_spec)
        
        # Log losses
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/td_loss', losses['td_loss'], on_step=True, on_epoch=True)
        self.log('train/sc_loss', losses['sc_loss'], on_step=True, on_epoch=True)
        self.log('train/mag_loss', losses['mag_loss'], on_step=True, on_epoch=True)
        self.log('train/mel_loss', losses['mel_loss'], on_step=True, on_epoch=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        mel_spec, target_audio, f0, lengths = self._process_batch(batch)
        
        # Forward pass
        generated_audio = self(mel_spec, f0=f0 if self.use_f0 else None)
        
        # Calculate losses
        losses = self._calculate_combined_loss(generated_audio, target_audio, mel_spec)
        
        # Log losses
        self.log('val/loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/td_loss', losses['td_loss'], on_step=False, on_epoch=True)
        self.log('val/sc_loss', losses['sc_loss'], on_step=False, on_epoch=True)
        self.log('val/mag_loss', losses['mag_loss'], on_step=False, on_epoch=True)
        self.log('val/mel_loss', losses['mel_loss'], on_step=False, on_epoch=True)
        
        # Log audio samples every N epochs
        if batch_idx == 0 and self.current_epoch % self.config['train'].get('log_vocoder_audio_epoch_interval', 5) == 0:
            self._log_audio_samples(mel_spec, generated_audio, target_audio)
            # Log mel comparison
            self._log_mel_comparison(mel_spec, generated_audio)
            
        # Return the dictionary of losses for potential callbacks or further analysis
        return losses
    
    def _log_audio_samples(self, mel_spec, generated_audio, target_audio):
        """
        Log audio samples and spectrograms for visualization
        """
        # Only log the first sample in the batch
        idx = 0
        
        # Get original mel spectrogram for visualization
        mel_to_plot = mel_spec[idx].detach().cpu().numpy().T  # [M, T]
        
        # Plot the mel spectrogram
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(mel_to_plot, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title("Input Mel Spectrogram")
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
            
            # Log audio samples - now at full audio rate
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
                
                # Log audio at full sample rate since our model now outputs at audio rate
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
        """
        Create and log a visualization comparing input mel spectrograms with 
        mel spectrograms generated from the predicted audio
        
        Args:
            input_mel (torch.Tensor): Input mel spectrogram [B, T, M]
            generated_audio (torch.Tensor): Generated audio from the model [B, T*hop_length, 1]
        """
        # Only use the first sample from the batch for visualization
        idx = 0
        
        # Get the input mel spectrogram
        input_mel_np = input_mel[idx].detach().cpu().numpy().T  # [M, T]
        
        # Get the expected time dimension for the generated mel
        expected_frames = input_mel_np.shape[1]
        
        # Convert generated audio back to mel spectrogram - now appropriately at full audio rate
        gen_audio_flat = generated_audio[idx, :, 0].detach().cpu().unsqueeze(0)  # [1, T*hop_length]
        
        # Use the initialized transform for consistency
        with torch.no_grad():
             gen_mel = self.mel_spectrogram_transform(gen_audio_flat.to(self.device)) # Ensure device match
             gen_mel_np = gen_mel[0].cpu().numpy() # Shape [M, T_mel_gen]
        
        # Get shapes for comparison and potential resizing
        input_shape = input_mel_np.shape # Should be [M, T_mel_target]
        gen_shape = gen_mel_np.shape   # Should be [M, T_mel_gen]
        # print(f"Input mel shape: {input_shape}, Generated mel shape: {gen_shape}") # Optional debug print
        
        # If lengths still don't match exactly, resize for visualization
        if gen_shape[1] != input_shape[1]:
            from scipy.ndimage import zoom
            # Get the scaling factor
            scale_factor = input_shape[1] / gen_shape[1]
            # Resize only the time dimension
            gen_mel_np = zoom(gen_mel_np, (1, scale_factor), order=1)
            if self.global_step % 100 == 0: # Log resizing less frequently
                 print(f"Resized generated mel from {gen_shape} to {gen_mel_np.shape} for visualization")
        
        # Create figure with two subplots vertically stacked
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