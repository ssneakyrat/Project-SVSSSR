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

import itertools
from models.hifigan.generator import Generator
from models.hifigan.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
# from models.unet_vocoder.utils import audio_to_mel # Keep if needed for mel loss
from utils.plotting import plot_spectrograms_to_figure

# GAN Loss Helper Functions (can be moved to a utils file later)
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2 # Multiply by 2 as in the official HiFi-GAN implementation

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses

class VocoderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Detach config from graph to avoid issues with saving hyperparameters
        self.config = config.copy()
        self.save_hyperparameters(self.config)
        self.automatic_optimization = False # Enable manual optimization
        
        # Initialize HiFi-GAN Generator and Discriminators
        self.generator = Generator(self.config)
        self.mpd = MultiPeriodDiscriminator(self.config)
        self.msd = MultiScaleDiscriminator(self.config)
        self.use_f0 = self.config['vocoder'].get('use_f0_conditioning', False) # Keep f0 flag
        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_fft = self.config['audio']['n_fft'] # Use n_fft from config
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio'].get('fmin', 0)
        self.fmax = self.config['audio'].get('fmax', self.sample_rate // 2)
        
        # Remove noise scale if not used by Generator directly
        # self.noise_scale = self.config['vocoder'].get('noise_scale', 0.6)
        
        # GAN Loss configuration (from config)
        self.lambda_fm = self.config['train'].get('lambda_feature_match', 2.0) # Weight for feature matching loss
        self.lambda_adv = self.config['train'].get('lambda_adversarial', 1.0) # Weight for adversarial loss (generator)
        self.lambda_mel_gan = self.config['train'].get('lambda_mel_gan', 45.0) # Weight for Mel Spectrogram loss (generator)
        
        # Keep Mel Spectrogram transform for potential Mel loss in Generator
        # (Ensure parameters match upstream model's mel generation)
        
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
        Forward pass through the HiFi-GAN Generator.

        Args:
            mel_spec (torch.Tensor): Mel spectrogram [B, T_mel, M]
            f0 (torch.Tensor, optional): Aligned F0 contour [B, T_mel, 1]. Required if use_f0 is True.

        Returns:
            torch.Tensor: Generated waveform [B, 1, T_audio]
        """
        # Reshape mel: [B, T_mel, M] -> [B, M, T_mel]
        mel_spec = mel_spec.transpose(1, 2)
        
        # Reshape f0 if needed: [B, T_mel, 1] -> [B, 1, T_mel]
        if f0 is not None:
            f0 = f0.transpose(1, 2)
            
        # Generate waveform with the HiFi-GAN Generator
        waveform = self.generator(mel_spec, f0=f0 if self.use_f0 else None) # Generator handles f0 internally if configured

        # Output shape: [B, 1, T_audio]
        return waveform
    
    def _process_batch(self, batch):
        """
        Process batch data to extract relevant inputs and targets
        """
        mel_spec = batch.get('mel_spec')
        target_audio = batch.get('target_audio')
        lengths = batch.get('length')  # Original audio lengths before padding
        f0 = batch.get(self.config['data'].get('f0_key', 'f0_contour'), None) # Get F0 using key from config
        
        # Reshape target audio if needed: [B, T_audio] -> [B, 1, T_audio]
        if target_audio is not None and target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(1)
        elif target_audio is not None and target_audio.dim() == 3 and target_audio.shape[-1] == 1:
             # If it's [B, T_audio, 1], transpose to [B, 1, T_audio]
             target_audio = target_audio.transpose(1, 2)
            
        # Reshape f0 if needed: [B, T_mel] -> [B, T_mel, 1]
        if f0 is not None and f0.dim() == 2:
            f0 = f0.unsqueeze(-1)

        # Ensure f0 is on the same device as mel_spec
        if f0 is not None and mel_spec is not None and f0.device != mel_spec.device:
             f0 = f0.to(mel_spec.device)
             
        return mel_spec, target_audio, f0, lengths
    
    # --- GAN Loss Calculation Methods ---

    def _calculate_discriminator_loss(self, target_audio, generated_audio):
        """ Calculates the loss for both MPD and MSD discriminators. """
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(target_audio, generated_audio.detach())
        loss_disc_f, losses_f_r, losses_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(target_audio, generated_audio.detach())
        loss_disc_s, losses_s_r, losses_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        
        loss_dict = {
            "loss_disc_all": loss_disc_all,
            "mpd_loss": loss_disc_f,
            "msd_loss": loss_disc_s,
            "mpd_real_losses": losses_f_r,
            "mpd_fake_losses": losses_f_g,
            "msd_real_losses": losses_s_r,
            "msd_fake_losses": losses_s_g,
        }
        return loss_dict

    def _calculate_generator_loss(self, target_audio, generated_audio):
        """ Calculates the loss for the generator. """
        # L1 Mel Loss
        # Ensure target_audio is [B, 1, T]
        if target_audio.dim() == 3 and target_audio.shape[1] != 1:
             target_audio_mel = target_audio.transpose(1, 2) # Assume [B, T, 1] -> [B, 1, T]
        else:
             target_audio_mel = target_audio
             
        # Ensure generated_audio is [B, 1, T]
        if generated_audio.dim() == 3 and generated_audio.shape[1] != 1:
             generated_audio_mel = generated_audio.transpose(1, 2)
        else:
             generated_audio_mel = generated_audio
             
        # Need to ensure lengths match before mel transform if they differ
        # For simplicity here, assume lengths are compatible or handled upstream/downstream
        # Trim target audio to match generated audio length for mel loss calculation
        min_len = min(target_audio_mel.shape[-1], generated_audio_mel.shape[-1])
        target_audio_mel = target_audio_mel[..., :min_len]
        generated_audio_mel = generated_audio_mel[..., :min_len]

        # Calculate mels - ensure transform is on the correct device
        self.mel_spectrogram_transform = self.mel_spectrogram_transform.to(generated_audio_mel.device)
        y_mel = self.mel_spectrogram_transform(target_audio_mel.squeeze(1)) # Input needs to be [B, T]
        y_g_hat_mel = self.mel_spectrogram_transform(generated_audio_mel.squeeze(1)) # Input needs to be [B, T]
        
        # Mel loss expects [B, M, T]
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * self.lambda_mel_gan

        # GAN Loss (Adversarial + Feature Matching)
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(target_audio, generated_audio)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(target_audio, generated_audio)
        
        # Feature Matching Loss
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_fm = (loss_fm_f + loss_fm_s) * self.lambda_fm

        # Adversarial Loss
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = (loss_gen_s + loss_gen_f) * self.lambda_adv

        # Total Generator Loss
        loss_gen_total = loss_gen_all + loss_fm + loss_mel
        
        loss_dict = {
            "loss_gen_total": loss_gen_total,
            "loss_gen_adv": loss_gen_all,
            "loss_gen_fm": loss_fm,
            "loss_gen_mel": loss_mel,
            "mpd_gen_losses": losses_gen_f,
            "msd_gen_losses": losses_gen_s,
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        mel_spec, target_audio, f0, lengths = self._process_batch(batch)
        opt_g, opt_d = self.optimizers() # Get optimizers for manual control
        
        # Ensure target_audio is [B, 1, T_audio]
        if target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(1)
        elif target_audio.dim() == 3 and target_audio.shape[-1] == 1:
            target_audio = target_audio.transpose(1, 2)

        # Generate audio
        # Input mel: [B, T_mel, M], f0: [B, T_mel, 1]
        # Output audio: [B, 1, T_audio]
        generated_audio = self(mel_spec, f0=f0 if self.use_f0 else None)

        # --- Train Discriminator ---
        # Detach generator output to avoid backprop through generator
        disc_losses = self._calculate_discriminator_loss(target_audio, generated_audio.detach())
        
        # Manual optimization for Discriminator
        opt_d.zero_grad()
        self.manual_backward(disc_losses['loss_disc_all'])
        opt_d.step()
        
        # Log discriminator losses
        self.log('train_disc/loss', disc_losses['loss_disc_all'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_disc/mpd_loss', disc_losses['mpd_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_disc/msd_loss', disc_losses['msd_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # Log individual period/scale losses if desired

        # --- Train Generator ---
        gen_losses = self._calculate_generator_loss(target_audio, generated_audio)

        # Manual optimization for Generator
        opt_g.zero_grad()
        self.manual_backward(gen_losses['loss_gen_total'])
        opt_g.step()

        # Log generator losses
        self.log('train_gen/loss', gen_losses['loss_gen_total'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_gen/adv_loss', gen_losses['loss_gen_adv'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_gen/fm_loss', gen_losses['loss_gen_fm'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_gen/mel_loss', gen_losses['loss_gen_mel'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # Log individual generator adversarial losses if desired
        
        # No return needed in manual optimization
    
    def validation_step(self, batch, batch_idx):
        mel_spec, target_audio, f0, lengths = self._process_batch(batch)
        
        # Ensure target_audio is [B, 1, T_audio]
        if target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(1)
        elif target_audio.dim() == 3 and target_audio.shape[-1] == 1:
            target_audio = target_audio.transpose(1, 2)
            
        # Forward pass
        generated_audio = self(mel_spec, f0=f0 if self.use_f0 else None)
        
        # Calculate Mel loss for validation
        # Trim target audio to match generated audio length
        min_len = min(target_audio.shape[-1], generated_audio.shape[-1])
        target_audio_val = target_audio[..., :min_len]
        generated_audio_val = generated_audio[..., :min_len]
        
        self.mel_spectrogram_transform = self.mel_spectrogram_transform.to(generated_audio_val.device)
        y_mel = self.mel_spectrogram_transform(target_audio_val.squeeze(1))
        y_g_hat_mel = self.mel_spectrogram_transform(generated_audio_val.squeeze(1))
        val_mel_loss = F.l1_loss(y_mel, y_g_hat_mel)
        
        # Log validation loss (using Mel loss as primary metric)
        self.log('val/mel_loss', val_mel_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store generated audio for logging if needed (use generated_audio, not generated_audio_val)
        # Note: _log_audio_samples and _log_mel_comparison expect [B, T, 1] or [B, T] format
        # Need to adjust them or the data passed to them.
        # For now, pass the original generated_audio and target_audio
        # Ensure target_audio is reshaped back if needed by logging functions
        if target_audio.dim() == 3 and target_audio.shape[1] == 1:
             target_audio_log = target_audio.transpose(1, 2) # [B, T, 1]
        else:
             target_audio_log = target_audio # Assume it's already correct or handled inside log func

        if generated_audio.dim() == 3 and generated_audio.shape[1] == 1:
             generated_audio_log = generated_audio.transpose(1, 2) # [B, T, 1]
        else:
             generated_audio_log = generated_audio
        
        # Log audio samples every N epochs
        if batch_idx == 0 and self.current_epoch % self.config['train'].get('log_vocoder_audio_epoch_interval', 5) == 0:
            self._log_audio_samples(mel_spec, generated_audio_log, target_audio_log)
            # Log mel comparison - needs generated audio in [B, T*hop, 1] format
            self._log_mel_comparison(mel_spec, generated_audio_log)
            
        # Return the dictionary of losses for potential callbacks or further analysis
        # Return validation loss for potential callbacks
        return {"val_mel_loss": val_mel_loss}
    
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
                # Ensure audio is [T]
                gen_audio = generated_audio[idx].squeeze().detach().cpu() # Remove channel dim
                tgt_audio = target_audio[idx].squeeze().detach().cpu() # Remove channel dim
                
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
            generated_audio (torch.Tensor): Generated audio from the model [B, T_audio, 1] or [B, 1, T_audio]
        """
        # Only use the first sample from the batch for visualization
        idx = 0
        
        # Get the input mel spectrogram
        input_mel_np = input_mel[idx].detach().cpu().numpy().T  # [M, T]
        
        # Get the expected time dimension for the generated mel
        expected_frames = input_mel_np.shape[1]
        
        # Convert generated audio back to mel spectrogram - now appropriately at full audio rate
        # Ensure input is [1, T_audio]
        if generated_audio.dim() == 3:
            gen_audio_flat = generated_audio[idx].squeeze(1).detach().cpu() # [1, T_audio] if [B, 1, T]
            if gen_audio_flat.dim() == 2 and gen_audio_flat.shape[0] != 1: # Check if it was [B, T, 1] -> [T, 1]
                gen_audio_flat = generated_audio[idx].squeeze().detach().cpu().unsqueeze(0) # [1, T_audio]
        else: # Assume [B, T]
            gen_audio_flat = generated_audio[idx].detach().cpu().unsqueeze(0)
        
        # Use the initialized transform for consistency
        with torch.no_grad():
            gen_mel = self.mel_spectrogram_transform(gen_audio_flat.to(self.device)) # Ensure device match
            gen_mel_np = gen_mel[0].cpu().numpy() # Shape should be [M, T_mel_gen]
            
            # Check if gen_mel_np is 1D and needs reshaping
            if len(gen_mel_np.shape) == 1:
                # Reshape 1D array into 2D - using n_mels from config
                num_mel_bins = self.n_mels
                # Calculate expected time frames
                time_frames = gen_mel_np.shape[0] // num_mel_bins
                if time_frames > 0:
                    # Reshape to (mel_bins, time_frames)
                    gen_mel_np = gen_mel_np[:num_mel_bins * time_frames].reshape(num_mel_bins, time_frames)
                else:
                    # If we can't determine the time frames, use the original mel shape as reference
                    time_frames = max(1, gen_mel_np.shape[0] // num_mel_bins)
                    gen_mel_np = gen_mel_np[:num_mel_bins * time_frames].reshape(num_mel_bins, time_frames)
                
                if self.global_step % 10 == 0:  # Log occasionally
                    print(f"Reshaped 1D mel spectrogram with shape {gen_mel_np.shape[0]} to 2D shape: {gen_mel_np.shape}")
        
        # Get shapes for comparison and potential resizing
        input_shape = input_mel_np.shape # Should be [M, T_mel_target]
        gen_shape = gen_mel_np.shape   # Should be [M, T_mel_gen]
        
        # Check if both shapes are valid (at least 2 dimensions) before comparing time dimension
        if len(gen_shape) >= 2 and len(input_shape) >= 2:
            # If lengths still don't match exactly, resize for visualization
            if gen_shape[1] != input_shape[1]:
                from scipy.ndimage import zoom
                # Get the scaling factor
                scale_factor = input_shape[1] / gen_shape[1]
                # Resize only the time dimension
                gen_mel_np = zoom(gen_mel_np, (1, scale_factor), order=1)
                if self.global_step % 100 == 0: # Log resizing less frequently
                    print(f"Resized generated mel from {gen_shape} to {gen_mel_np.shape} for visualization")
        elif self.global_step % 10 == 0: # Log error more frequently if shapes are wrong
            print(f"Warning: Mel shapes are incompatible for comparison/resizing. Input: {input_shape}, Generated: {gen_shape}. Skipping resizing.")
        
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
        # Get learning rates and betas from config
        lr_g = self.config['train'].get('generator_learning_rate', 0.0002)
        lr_d = self.config['train'].get('discriminator_learning_rate', 0.0002)
        betas_g = tuple(self.config['train'].get('generator_betas', [0.8, 0.99]))
        betas_d = tuple(self.config['train'].get('discriminator_betas', [0.8, 0.99]))
        weight_decay_g = self.config['train'].get('generator_weight_decay', 0.0)
        weight_decay_d = self.config['train'].get('discriminator_weight_decay', 0.0)
        
        # Optimizer for Generator
        optimizer_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr_g,
            betas=betas_g,
            weight_decay=weight_decay_g
        )
        
        # Optimizer for Discriminators (MPD + MSD)
        optimizer_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr=lr_d,
            betas=betas_d,
            weight_decay=weight_decay_d
        )

        # Schedulers (Example: ExponentialLR, common for HiFi-GAN)
        scheduler_gamma = self.config['train'].get('lr_scheduler_gamma', 0.999)
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=scheduler_gamma)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=scheduler_gamma)

        return (
            {'optimizer': optimizer_g, 'lr_scheduler': {'scheduler': scheduler_g, 'interval': 'epoch'}},
            {'optimizer': optimizer_d, 'lr_scheduler': {'scheduler': scheduler_d, 'interval': 'epoch'}}
        )