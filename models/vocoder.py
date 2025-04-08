import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import io
from PIL import Image
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from models.unet_vocoder.model import UNetVocoder
from models.unet_vocoder.utils import generate_noise, audio_to_mel
from utils.plotting import plot_spectrograms_to_figure

class VocoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize the vocoder model
        self.vocoder = UNetVocoder(config)
        
        # Audio parameters
        self.sample_rate = config['audio']['sample_rate']
        self.hop_length = config['audio']['hop_length']
        self.win_length = config['audio']['win_length']
        self.n_mels = config['audio']['n_mels']
        
        # Store noise scale for inference
        self.noise_scale = config['vocoder'].get('noise_scale', 0.6)
        
        # Loss weights
        self.loss_type = config['train'].get('vocoder_loss', 'L1')
        self.lambda_td = config['train'].get('loss_lambda_td', 1.0)  # Time-domain loss weight
        
    def forward(self, mel_spec):
        """
        Forward pass through the vocoder model
        
        Args:
            mel_spec (torch.Tensor): Mel spectrogram [B, T, M]
            
        Returns:
            torch.Tensor: Generated waveform [B, T*hop_length, 1]
        """
        batch_size, time_steps, _ = mel_spec.size()
        
        # Generate noise with the SAME time dimension as mel_spec
        noise = torch.randn(batch_size, time_steps, 1, device=mel_spec.device) * self.noise_scale
        
        # Generate waveform with the UNetVocoder
        # The UNetVocoder now handles the internal upsampling to audio rate
        waveform = self.vocoder(mel_spec, noise)
        
        return waveform
    
    def _process_batch(self, batch):
        """
        Process batch data to extract relevant inputs and targets
        """
        mel_spec = batch.get('mel_spec')
        target_audio = batch.get('target_audio')
        lengths = batch.get('length')  # Original lengths before padding
        
        # Reshape target audio if needed
        if target_audio.dim() == 2:  # [B, T]
            target_audio = target_audio.unsqueeze(-1)  # [B, T, 1]
            
        return mel_spec, target_audio, lengths
    
    def time_domain_loss(self, y_pred, y_true):
        """
        Compute time-domain loss between predicted and target waveforms
        
        Args:
            y_pred: Predicted waveform from model [B, T*hop_length, 1]
            y_true: Target waveform from dataset [B, T*hop_length, 1]
            
        Returns:
            Loss value
        """
        # Check if the shapes match exactly
        if y_pred.shape[1] == y_true.shape[1]:
            return F.l1_loss(y_pred, y_true)
            
        # If shapes don't match exactly, we may need to handle it
        # Get the minimum length and truncate both to that length
        min_length = min(y_pred.shape[1], y_true.shape[1])
        y_pred_trimmed = y_pred[:, :min_length, :]
        y_true_trimmed = y_true[:, :min_length, :]
        
        # Log the length mismatch
        if self.training and hasattr(self, 'global_step') and self.global_step % 100 == 0:
            print(f"Warning: Length mismatch in time_domain_loss. "
                  f"Pred: {y_pred.shape[1]}, Target: {y_true.shape[1]}, "
                  f"Using first {min_length} samples")
        
        return F.l1_loss(y_pred_trimmed, y_true_trimmed)
    
    def training_step(self, batch, batch_idx):
        mel_spec, target_audio, lengths = self._process_batch(batch)
        
        # Forward pass
        generated_audio = self(mel_spec)
        
        # Calculate time-domain loss
        td_loss = self.time_domain_loss(generated_audio, target_audio)
        
        # Apply weight
        loss = self.lambda_td * td_loss
        
        # Log losses
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/td_loss', td_loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        mel_spec, target_audio, lengths = self._process_batch(batch)
        
        # Forward pass
        generated_audio = self(mel_spec)
        
        # Calculate time-domain loss
        td_loss = self.time_domain_loss(generated_audio, target_audio)
        
        # Apply weight
        loss = self.lambda_td * td_loss
        
        # Log losses
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.log('val/td_loss', td_loss, on_step=False, on_epoch=True)
        
        # Log audio samples every N epochs
        if batch_idx == 0 and self.current_epoch % self.config['train'].get('log_vocoder_audio_epoch_interval', 5) == 0:
            self._log_audio_samples(mel_spec, generated_audio, target_audio)
            # Log mel comparison
            self._log_mel_comparison(mel_spec, generated_audio)
            
        return loss
    
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
        
        # Create a config dict with just the required audio parameters
        audio_config = {
            'audio': {
                'sample_rate': self.sample_rate,
                'n_fft': self.win_length,  # Assuming n_fft = win_length
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'n_mels': self.n_mels,
                'fmin': 0,  # Using default values
                'fmax': self.sample_rate // 2  # Using Nyquist frequency
            }
        }
        
        # Convert generated audio to mel spectrogram
        gen_mel = audio_to_mel(gen_audio_flat, audio_config)
        gen_mel_np = gen_mel[0].numpy().T  # [M, T]
        
        # Print shape information for debugging
        input_shape = input_mel_np.shape
        gen_shape = gen_mel_np.shape
        print(f"Input mel shape: {input_shape}, Generated mel shape: {gen_shape}")
        
        # If lengths still don't match exactly, resize for visualization
        if gen_shape[1] != input_shape[1]:
            from scipy.ndimage import zoom
            # Get the scaling factor
            scale_factor = input_shape[1] / gen_shape[1]
            # Resize only the time dimension
            gen_mel_np = zoom(gen_mel_np, (1, scale_factor), order=1)
            print(f"Resized generated mel to {gen_mel_np.shape} for visualization")
        
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