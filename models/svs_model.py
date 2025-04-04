import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

from models.vae import VAE
from models.diffusion import DiffusionModel
from models.conditioning import ConditioningEncoder, FeatureAligner


class SVSModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create VAE for latent space compression
        self.vae = VAE(config)
        
        # Create conditioning encoder
        self.conditioning_encoder = ConditioningEncoder(config)
        
        # Create feature aligner to match conditioning time dimension to latent space
        # Calculate latent dimensions based on the VAE architecture
        vae_downsample_factor = 2 ** len(config['model']['vae']['encoder_channels'])  # 2^3 = 8x reduction
        mel_time_frames = config['model']['time_frames']
        latent_time_frames = mel_time_frames // vae_downsample_factor
        mel_bins = config['model']['mel_bins']
        latent_bins = mel_bins // vae_downsample_factor
        
        print(f"Latent dimensions: time={latent_time_frames}, freq={latent_bins}")
        
        self.feature_aligner = FeatureAligner(
            in_channels=config['model']['conditioning']['condition_channels'],
            out_channels=config['model']['conditioning']['condition_channels'],
            out_time_dim=latent_time_frames
        )
        
        # Save dimensions for inference
        self.latent_time_frames = latent_time_frames
        self.latent_bins = latent_bins
        self.vae_downsample_factor = vae_downsample_factor
        
        # Create diffusion model
        self.diffusion = DiffusionModel(config)
        
        # Track losses for logging
        self.vae_weight = 1.0
        self.diffusion_weight = 1.0
        
    def encode_conditioning(self, f0, phone, duration, midi):
        """
        Encode conditioning inputs
        
        Args:
            f0: F0 contour [batch, time_steps]
            phone: Phone labels [batch, time_steps]
            duration: Phone durations [batch, time_steps]
            midi: MIDI note labels [batch, time_steps]
            
        Returns:
            Conditioning tensor aligned to latent dimensions [batch, latent_time, channels]
        """
        # Process all conditioning inputs
        conditioning = self.conditioning_encoder(f0, phone, duration, midi)
        
        # Align to latent space time dimension
        latent_conditioning = self.feature_aligner(conditioning)
        
        return latent_conditioning
    
    def forward(self, mel, f0, phone, duration, midi):
        """
        Full forward pass for training
        
        Args:
            mel: Mel spectrogram [batch, 1, freq_bins, time_frames]
            f0: F0 contour [batch, time_steps]
            phone: Phone labels [batch, time_steps]
            duration: Phone durations [batch, time_steps]
            midi: MIDI note labels [batch, time_steps]
            
        Returns:
            Dictionary with loss values and intermediate outputs
        """
        # Encode conditioning
        conditioning = self.encode_conditioning(f0, phone, duration, midi)
        
        # Encode mel to latent space
        latent, mu, log_var = self.vae.encode(mel)
        
        # Train diffusion in latent space
        diffusion_loss = self.diffusion(latent, conditioning)
        
        # Also train VAE reconstruction
        mel_recon, _, vae_loss, recon_loss, kl_loss = self.vae(mel)
        
        # Total loss
        total_loss = (
            self.vae_weight * vae_loss + 
            self.diffusion_weight * diffusion_loss
        )
        
        return {
            'loss': total_loss,
            'vae_loss': vae_loss,
            'diffusion_loss': diffusion_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mel_recon': mel_recon,
            'latent': latent
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Dictionary with input tensors
            batch_idx: Batch index
            
        Returns:
            Loss scalar
        """
        # Extract inputs from batch
        mel = batch['mel']
        f0 = batch['f0']
        phone = batch['phone'] 
        duration = batch['duration']
        midi = batch['midi']
        
        # Forward pass
        outputs = self.forward(mel, f0, phone, duration, midi)
        
        # Log losses
        self.log('train_loss', outputs['loss'], prog_bar=True)
        self.log('train_vae_loss', outputs['vae_loss'])
        self.log('train_diffusion_loss', outputs['diffusion_loss'])
        self.log('train_recon_loss', outputs['recon_loss'])
        self.log('train_kl_loss', outputs['kl_loss'])
        
        return outputs['loss']
    
    def _log_diffusion_samples(self, original, f0, phone, duration, midi):
        """
        Generate and log samples from the diffusion model during validation
        
        Args:
            original: Original mel spectrograms [batch, 1, freq_bins, time_frames]
            f0: F0 contour [batch, time_steps]
            phone: Phone labels [batch, time_steps]
            duration: Phone durations [batch, time_steps]
            midi: MIDI note labels [batch, time_steps]
        """
        # Take first 4 samples from batch
        num_samples = min(4, original.size(0))
        
        # Generate samples with the diffusion model
        with torch.no_grad():
            # Generate mel using just these samples
            diffusion_output = self.infer(
                f0[:num_samples], 
                phone[:num_samples], 
                duration[:num_samples], 
                midi[:num_samples]
            )
        
        for i in range(num_samples):
            # Convert to numpy for plotting
            orig_mel = original[i, 0].detach().cpu().numpy()
            diff_mel = diffusion_output[i, 0].detach().cpu().numpy()
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            # Plot original
            im1 = ax1.imshow(orig_mel, aspect='auto', origin='lower', interpolation='none')
            ax1.set_title("Original Mel-Spectrogram")
            plt.colorbar(im1, ax=ax1)
            
            # Plot diffusion generated output
            im2 = ax2.imshow(diff_mel, aspect='auto', origin='lower', interpolation='none')
            ax2.set_title("Diffusion Generated Mel-Spectrogram")
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            # Convert to image
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to tensor
            image = transforms.ToTensor()(Image.open(buf))
            
            # Log to tensorboard
            self.logger.experiment.add_image(f'diffusion_sample_{i}', image, self.global_step)
            
            plt.close(fig)
            
        # Also create a comparison plot showing original, VAE reconstruction and diffusion output
        for i in range(num_samples):
            # Get VAE reconstruction by encoding and decoding the original
            with torch.no_grad():
                latent, _, _ = self.vae.encode(original[i:i+1])
                vae_recon = self.vae.decode(latent)
                vae_recon_mel = vae_recon[0, 0].detach().cpu().numpy()
                
            # Convert to numpy for plotting
            orig_mel = original[i, 0].detach().cpu().numpy()
            diff_mel = diffusion_output[i, 0].detach().cpu().numpy()
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
            
            # Plot original
            im1 = ax1.imshow(orig_mel, aspect='auto', origin='lower', interpolation='none')
            ax1.set_title("Original Mel-Spectrogram")
            plt.colorbar(im1, ax=ax1)
            
            # Plot VAE reconstruction
            im2 = ax2.imshow(vae_recon_mel, aspect='auto', origin='lower', interpolation='none')
            ax2.set_title("VAE Reconstruction")
            plt.colorbar(im2, ax=ax2)
            
            # Plot diffusion generated output
            im3 = ax3.imshow(diff_mel, aspect='auto', origin='lower', interpolation='none')
            ax3.set_title("Diffusion Generated (Conditioned)")
            plt.colorbar(im3, ax=ax3)
            
            plt.tight_layout()
            
            # Convert to image
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to tensor
            image = transforms.ToTensor()(Image.open(buf))
            
            # Log to tensorboard
            self.logger.experiment.add_image(f'comparison_{i}', image, self.global_step)
            
            plt.close(fig)
            
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Dictionary with input tensors
            batch_idx: Batch index
            
        Returns:
            Loss scalar
        """
        # Extract inputs from batch
        mel = batch['mel']
        f0 = batch['f0']
        phone = batch['phone']
        duration = batch['duration']
        midi = batch['midi']
        
        # Forward pass
        outputs = self.forward(mel, f0, phone, duration, midi)
        
        # Log losses
        self.log('val_loss', outputs['loss'], prog_bar=True)
        self.log('val_vae_loss', outputs['vae_loss'])
        self.log('val_diffusion_loss', outputs['diffusion_loss'])
        self.log('val_recon_loss', outputs['recon_loss'])
        self.log('val_kl_loss', outputs['kl_loss'])
        
        # Log sample reconstructions and diffusion outputs for first batch
        #if batch_idx == 0:
        if self.current_epoch % 5 == 0:
            self._log_reconstructions(mel, outputs['mel_recon'])
            
        # Only run diffusion visualization every 5 epochs to save time
        #if self.current_epoch % 5 == 0:
        #    self._log_diffusion_samples(mel, f0, phone, duration, midi)
            
        return outputs['loss']
    
    def infer(self, f0, phone, duration, midi):
        """
        Model inference - generate mel spectrogram
        
        Args:
            f0: F0 contour [batch, time_steps]
            phone: Phone labels [batch, time_steps]
            duration: Phone durations [batch, time_steps]
            midi: MIDI note labels [batch, time_steps]
            
        Returns:
            Generated mel spectrogram [batch, 1, freq_bins, time_frames]
        """
        # Encode conditioning
        conditioning = self.encode_conditioning(f0, phone, duration, midi)
        
        # Get latent shape based on conditioning time dimension and stored dimensions
        batch_size = conditioning.shape[0]
        latent_channels = self.config['model']['vae']['latent_channels']
        
        # Use the properly calculated dimensions for latent space
        latent_shape = (batch_size, latent_channels, self.latent_bins, self.latent_time_frames)
        
        # Sample from diffusion model
        latent = self.diffusion.p_sample_loop(latent_shape, conditioning)
        
        # Decode latent to get mel spectrogram
        mel = self.vae.decode(latent)
        
        return mel
    
    def _log_reconstructions(self, original, reconstructed):
        """
        Log sample reconstructions as images
        
        Args:
            original: Original mel spectrograms [batch, 1, freq_bins, time_frames]
            reconstructed: Reconstructed mel spectrograms [batch, 1, freq_bins, time_frames]
        """
        # Take first 4 samples from batch
        num_samples = min(4, original.size(0))
        
        for i in range(num_samples):
            # Convert to numpy for plotting
            orig_mel = original[i, 0].detach().cpu().numpy()
            recon_mel = reconstructed[i, 0].detach().cpu().numpy()
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            # Plot original
            im1 = ax1.imshow(orig_mel, aspect='auto', origin='lower', interpolation='none')
            ax1.set_title("Original Mel-Spectrogram")
            plt.colorbar(im1, ax=ax1)
            
            # Plot reconstruction
            im2 = ax2.imshow(recon_mel, aspect='auto', origin='lower', interpolation='none')
            ax2.set_title("Reconstructed Mel-Spectrogram")
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            # Convert to image
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to tensor
            image = transforms.ToTensor()(Image.open(buf))
            
            # Log to tensorboard
            self.logger.experiment.add_image(f'reconstruction_{i}', image, self.global_step)
            
            plt.close(fig)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers
        
        Returns:
            Optimizer and scheduler configuration
        """
        lr = self.config['train']['learning_rate']
        weight_decay = self.config['train']['weight_decay']
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Create learning rate scheduler
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config['train']['lr_factor'],
                patience=self.config['train']['lr_patience'],
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            # Step LR scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }