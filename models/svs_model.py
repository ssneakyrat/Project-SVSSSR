import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple, Optional, List, Union

from models.base_model import BaseSVSModel
from models.conditioning import ConditioningEncoder, FeatureAligner
from models.vae import VAE
from models.diffusion import DiffusionModel


class SVSModel(BaseSVSModel):
    """
    Main SVS model that integrates conditioning, VAE, and diffusion modules.
    Supports separate or joint training modes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get configuration
        self.model_config = config["model"]
        self.train_config = config["training"]
        self.mode = self.train_config["mode"]
        
        # Create conditioning encoder
        self.conditioning_encoder = ConditioningEncoder(self.model_config["conditioning"])
        
        # Calculate latent dimensions
        self.setup_dimensions()
        
        # Create feature aligner for conditioning -> latent time dimension
        self.feature_aligner = FeatureAligner({
            "in_channels": self.model_config["conditioning"]["hidden_channels"],
            "out_channels": self.model_config["conditioning"]["hidden_channels"],
            "target_length": self.latent_time_frames,
            "dropout": self.model_config["conditioning"].get("dropout", 0.1)
        })
        
        # Create VAE
        self.vae = VAE(self.model_config["vae"])
        
        # Create diffusion model
        diffusion_config = self.model_config["diffusion"]
        diffusion_config["in_channels"] = self.model_config["vae"]["latent_channels"]
        diffusion_config["context_dim"] = self.model_config["conditioning"]["hidden_channels"]
        self.diffusion = DiffusionModel(diffusion_config)
        
        # Set loss weights
        self.vae_weight = self.train_config["joint_training"]["vae_weight"]
        self.diffusion_weight = self.train_config["joint_training"]["diffusion_weight"]
        
        # Set mode-specific configurations
        self.setup_training_mode()
        
    def setup_dimensions(self):
        """Setup latent dimensions based on model config."""
        # Get mel dimensions
        mel_bins = self.model_config.get("mel_bins", 80)
        time_frames = self.model_config.get("max_frames", 432)  # ~5 seconds at 22050Hz/256 hop
        
        # Calculate downsampling factor from VAE config
        vae_encoder_channels = self.model_config["vae"]["encoder_channels"]
        # Each downsampling reduces dimensions by factor of 2
        downsample_factor = 2 ** (len(vae_encoder_channels) - 1)
        
        # Calculate latent dimensions
        self.latent_bins = mel_bins // downsample_factor
        self.latent_time_frames = time_frames // downsample_factor
        
        print(f"Mel dimensions: {mel_bins}x{time_frames}")
        print(f"Latent dimensions: {self.latent_bins}x{self.latent_time_frames}")
    
    def setup_training_mode(self):
        """Configure the model for specified training mode."""
        if self.mode == "vae_only":
            # Freeze diffusion model
            for param in self.diffusion.parameters():
                param.requires_grad = False
                
            # Set specific learning rate for VAE
            self.learning_rate = self.train_config["vae_training"]["learning_rate"]
            
        elif self.mode == "diffusion_only":
            # Freeze VAE
            if self.train_config["diffusion_training"]["freeze_vae"]:
                for param in self.vae.parameters():
                    param.requires_grad = False
            
            # Set specific learning rate for diffusion
            self.learning_rate = self.train_config["diffusion_training"]["learning_rate"]
            
            # Set conditioning dropout probability
            self.condition_dropout = self.train_config["diffusion_training"]["condition_dropout"]
            
        else:  # joint training
            # Use default learning rate
            self.learning_rate = self.train_config["optimizer"]["learning_rate"]
            self.condition_dropout = 0.0
            
    def encode_conditioning(
        self, 
        f0: torch.Tensor, 
        phone: torch.Tensor, 
        duration: torch.Tensor, 
        midi: torch.Tensor
    ) -> torch.Tensor:
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
    
    def forward(
        self, 
        mel: torch.Tensor, 
        f0: torch.Tensor, 
        phone: torch.Tensor, 
        duration: torch.Tensor, 
        midi: torch.Tensor
    ) -> Dict[str, Any]:
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
        
        # Initialize output dictionary
        outputs = {}
        total_loss = 0.0
        
        # VAE forward pass
        if self.mode in ["vae_only", "joint"]:
            # Run VAE with current global step for KL annealing
            mel_recon, latent, vae_loss, recon_loss, kl_loss = self.vae(
                mel, global_step=self.global_step
            )
            
            # Add VAE outputs to dictionary
            outputs["mel_recon"] = mel_recon
            outputs["latent"] = latent
            outputs["vae_loss"] = vae_loss
            outputs["recon_loss"] = recon_loss
            outputs["kl_loss"] = kl_loss
            
            # Add to total loss with weight
            total_loss += self.vae_weight * vae_loss
        else:
            # In diffusion_only mode, just encode without computing loss
            latent, _, _ = self.vae.encode(mel)
            outputs["latent"] = latent
        
        # Diffusion forward pass
        if self.mode in ["diffusion_only", "joint"]:
            # Run diffusion with conditioning
            diffusion_loss, diffusion_metrics = self.diffusion(
                latent.detach() if self.mode == "diffusion_only" else latent,
                conditioning,
                condition_dropout_prob=self.condition_dropout if self.mode == "diffusion_only" else 0.0
            )
            
            # Add diffusion outputs to dictionary
            outputs["diffusion_loss"] = diffusion_loss
            outputs["diffusion_metrics"] = diffusion_metrics
            
            # Add to total loss with weight
            total_loss += self.diffusion_weight * diffusion_loss
        
        # Set total loss
        outputs["loss"] = total_loss
        
        return outputs
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step
        
        Args:
            batch: Dictionary with input tensors
            batch_idx: Batch index
            
        Returns:
            Loss scalar
        """
        # Extract inputs from batch
        mel = batch["mel"]
        f0 = batch["f0"]
        phone = batch["phone"] 
        duration = batch["duration"]
        midi = batch["midi"]
        
        # Forward pass
        outputs = self.forward(mel, f0, phone, duration, midi)
        
        # Log mode-specific losses
        if self.mode in ["vae_only", "joint"]:
            self.log("train/vae_loss", outputs["vae_loss"], prog_bar=True)
            self.log("train/recon_loss", outputs["recon_loss"])
            self.log("train/kl_loss", outputs["kl_loss"])
            
            # Log current KL weight
            if hasattr(self.vae, "kl_weight"):
                self.log("train/kl_weight", self.vae.kl_weight)
        
        if self.mode in ["diffusion_only", "joint"]:
            self.log("train/diffusion_loss", outputs["diffusion_loss"], prog_bar=True)
            
            # Log diffusion metrics
            for key, value in outputs["diffusion_metrics"].items():
                if isinstance(value, (int, float)):
                    self.log(f"train/diffusion_{key}", value)
        
        # Log total loss
        self.log("train/loss", outputs["loss"], prog_bar=True)
        
        return outputs["loss"]
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step
        
        Args:
            batch: Dictionary with input tensors
            batch_idx: Batch index
            
        Returns:
            Loss scalar
        """
        # Extract inputs from batch
        mel = batch["mel"]
        f0 = batch["f0"]
        phone = batch["phone"]
        duration = batch["duration"]
        midi = batch["midi"]
        
        # Forward pass
        outputs = self.forward(mel, f0, phone, duration, midi)
        
        # Log mode-specific losses
        if self.mode in ["vae_only", "joint"]:
            self.log("val/vae_loss", outputs["vae_loss"], prog_bar=True)
            self.log("val/recon_loss", outputs["recon_loss"])
            self.log("val/kl_loss", outputs["kl_loss"])
        
        if self.mode in ["diffusion_only", "joint"]:
            self.log("val/diffusion_loss", outputs["diffusion_loss"], prog_bar=True)
            
            # Log diffusion metrics
            for key, value in outputs["diffusion_metrics"].items():
                if isinstance(value, (int, float)):
                    self.log(f"val/diffusion_{key}", value)
        
        # Log total loss
        self.log("val/loss", outputs["loss"], prog_bar=True)
        
        # Log visualizations for first batch only
        if batch_idx == 0:
            # Log VAE reconstructions if applicable
            if self.mode in ["vae_only", "joint"] and "mel_recon" in outputs:
                self._log_reconstructions(mel, outputs["mel_recon"])
            
            # Log diffusion samples if applicable
            if self.mode in ["diffusion_only", "joint"]:
                self._log_diffusion_samples(mel, f0, phone, duration, midi)
        
        return outputs["loss"]
    
    def on_validation_epoch_end(self):
        """Called at the end of validation."""
        # Additional end-of-epoch validation visualizations can be added here
        pass
    
    def _log_reconstructions(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ):
        """
        Log sample reconstructions as images
        
        Args:
            original: Original mel spectrograms [batch, 1, freq_bins, time_frames]
            reconstructed: Reconstructed mel spectrograms [batch, 1, freq_bins, time_frames]
        """
        # Take first few samples from batch
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
            self.logger.experiment.add_image(f'vae_reconstruction_{i}', image, self.global_step)
            
            plt.close(fig)
    
    def _log_diffusion_samples(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor, 
        phone: torch.Tensor, 
        duration: torch.Tensor, 
        midi: torch.Tensor
    ):
        """
        Log diffusion-generated samples as images
        
        Args:
            mel: Original mel spectrograms [batch, 1, freq_bins, time_frames]
            f0, phone, duration, midi: Conditioning inputs
        """
        # Generate samples using a subset of the batch
        num_samples = min(2, mel.size(0))
        
        with torch.no_grad():
            # Get batch subset
            mel_subset = mel[:num_samples]
            f0_subset = f0[:num_samples]
            phone_subset = phone[:num_samples]
            duration_subset = duration[:num_samples]
            midi_subset = midi[:num_samples]
            
            # Encode conditioning
            conditioning = self.encode_conditioning(
                f0_subset, phone_subset, duration_subset, midi_subset
            )
            
            # Calculate latent shape
            latent_shape = (
                num_samples, 
                self.model_config["vae"]["latent_channels"],
                self.latent_bins,
                self.latent_time_frames
            )
            
            # Generate latents using diffusion model
            generated_latents = self.diffusion.p_sample_loop(
                latent_shape, 
                conditioning,
                progress=False
            )
            
            # Get ground truth latents for comparison
            true_latents, _, _ = self.vae.encode(mel_subset)
            
            # Decode both to get mel spectrograms
            generated_mels = self.vae.decode(generated_latents)
            reconstructed_mels = self.vae.decode(true_latents)
            
            # Plot comparisons
            for i in range(num_samples):
                # Get individual spectrograms
                orig_mel = mel_subset[i, 0].detach().cpu().numpy()
                recon_mel = reconstructed_mels[i, 0].detach().cpu().numpy()
                gen_mel = generated_mels[i, 0].detach().cpu().numpy()
                
                # Create figure with three subplots
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
                
                # Plot original
                im1 = ax1.imshow(orig_mel, aspect='auto', origin='lower', interpolation='none')
                ax1.set_title("Original Mel-Spectrogram")
                plt.colorbar(im1, ax=ax1)
                
                # Plot VAE reconstruction
                im2 = ax2.imshow(recon_mel, aspect='auto', origin='lower', interpolation='none')
                ax2.set_title("VAE Reconstruction")
                plt.colorbar(im2, ax=ax2)
                
                # Plot diffusion generation
                im3 = ax3.imshow(gen_mel, aspect='auto', origin='lower', interpolation='none')
                ax3.set_title("Diffusion Generated")
                plt.colorbar(im3, ax=ax3)
                
                plt.tight_layout()
                
                # Convert to image
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert to tensor
                image = transforms.ToTensor()(Image.open(buf))
                
                # Log to tensorboard
                self.logger.experiment.add_image(
                    f'diffusion_generation_{i}', 
                    image, 
                    self.global_step
                )
                
                plt.close(fig)
    
    def infer(
        self,
        f0: torch.Tensor, 
        phone: torch.Tensor, 
        duration: torch.Tensor, 
        midi: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Model inference - generate mel spectrogram from conditioning
        
        Args:
            f0: F0 contour [batch, time_steps]
            phone: Phone labels [batch, time_steps]
            duration: Phone durations [batch, time_steps]
            midi: MIDI note labels [batch, time_steps]
            temperature: Sampling temperature (noise scale)
            
        Returns:
            Generated mel spectrogram [batch, 1, freq_bins, time_frames]
        """
        with torch.no_grad():
            # Encode conditioning
            conditioning = self.encode_conditioning(f0, phone, duration, midi)
            
            # Get latent shape based on batch size
            batch_size = conditioning.shape[0]
            latent_shape = (
                batch_size, 
                self.model_config["vae"]["latent_channels"],
                self.latent_bins,
                self.latent_time_frames
            )
            
            # Sample from diffusion model
            noise = torch.randn(latent_shape, device=self.device) * temperature
            latent = self.diffusion.p_sample_loop(latent_shape, conditioning, noise=noise)
            
            # Decode latent to get mel spectrogram
            mel = self.vae.decode(latent)
            
            return mel
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers based on training mode
        
        Returns:
            Optimizer or tuple of (optimizer, scheduler)
        """
        # Get base optimizer config
        opt_config = self.train_config["optimizer"]
        
        # Create parameter groups based on training mode
        if self.mode == "vae_only":
            # Only optimize VAE and conditioning encoder
            params = list(self.vae.parameters()) + list(self.conditioning_encoder.parameters()) + list(self.feature_aligner.parameters())
            lr = self.train_config["vae_training"]["learning_rate"]
        elif self.mode == "diffusion_only":
            # Only optimize diffusion model and conditioning encoder
            params = list(self.diffusion.parameters()) + list(self.conditioning_encoder.parameters()) + list(self.feature_aligner.parameters())
            lr = self.train_config["diffusion_training"]["learning_rate"]
        else:  # joint training
            # Optimize all parameters but with different learning rates
            vae_params = list(self.vae.parameters())
            diffusion_params = list(self.diffusion.parameters())
            conditioning_params = list(self.conditioning_encoder.parameters()) + list(self.feature_aligner.parameters())
            
            # Create parameter groups with different learning rates
            params = [
                {"params": vae_params, "lr": opt_config["learning_rate"] * 0.5},  # Lower LR for VAE
                {"params": diffusion_params, "lr": opt_config["learning_rate"]},
                {"params": conditioning_params, "lr": opt_config["learning_rate"]}
            ]
            
            lr = opt_config["learning_rate"]  # Base learning rate
        
        # Create optimizer
        if opt_config["name"] == "adam":
            optimizer = torch.optim.Adam(
                params, 
                lr=lr,
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"]
            )
        elif opt_config["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                params, 
                lr=lr,
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
        
        # Get scheduler config
        sched_config = self.train_config["scheduler"]
        
        # Create scheduler
        if sched_config["name"] == "none":
            return optimizer
        
        if sched_config["name"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=sched_config["decay_steps"],
                eta_min=lr * sched_config["min_lr_ratio"]
            )
        elif sched_config["name"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config["decay_steps"] // 3,  # 3 steps over the training
                gamma=sched_config["min_lr_ratio"] ** (1/3)
            )
        elif sched_config["name"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=sched_config["decay_steps"] // 10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['name']}")
        
        # Wrap with warmup
        if sched_config["warmup_steps"] > 0:
            from transformers import get_cosine_schedule_with_warmup
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=sched_config["warmup_steps"],
                num_training_steps=sched_config["decay_steps"]
            )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]