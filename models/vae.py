import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from models.base_model import BaseModule


class SpectralNorm:
    """Spectral normalization for convolutional layers."""
    
    @staticmethod
    def apply(module, name='weight', n_power_iterations=1, eps=1e-12, dim=0):
        """Apply spectral normalization to a parameter in the module."""
        fn = SpectralNorm(name, n_power_iterations, eps, dim)
        weight = getattr(module, name)
        
        # Remove the parameter from the module parameters
        del module._parameters[name]
        
        # Add the new parameters with spectral normalization
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        setattr(module, name, weight.data)
        
        # Add weight to the module with forward hook to update weight
        module.register_forward_pre_hook(fn)
        
        # Initialize u and v vectors
        fn.create_vectors(module)
        
        return module
    
    def __init__(self, name='weight', n_power_iterations=1, eps=1e-12, dim=0):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.dim = dim
    
    def create_vectors(self, module):
        """Initialize u and v vectors for power iteration."""
        weight = getattr(module, self.name + "_orig")
        height = weight.size(self.dim)
        
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=self.eps)
        v = F.normalize(weight.new_empty(weight.size()).normal_(0, 1), dim=0, eps=self.eps)
        
        module.register_buffer(self.name + "_u", u)
        module.register_buffer(self.name + "_v", v)
    
    def reshape_weight_to_matrix(self, weight):
        """Reshape the weight tensor to a matrix for power iteration."""
        if self.dim == 0:
            return weight
        return weight.transpose(0, self.dim).reshape(weight.size(self.dim), -1)
    
    def __call__(self, module, inputs):
        """Update the weights using power iteration method."""
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        
        # Reshape the weight for power iteration
        weight_mat = self.reshape_weight_to_matrix(weight)
        
        with torch.no_grad():
            # Power iteration
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat, u.unsqueeze(1)).squeeze(1), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat.t(), v.unsqueeze(1)).squeeze(1), dim=0, eps=self.eps)
            
            # Calculate sigma (the spectral norm)
            sigma = torch.dot(u, torch.matmul(weight_mat, v))
        
        # Update the weight for forward pass
        setattr(module, self.name, weight / sigma)


class ResidualBlock(nn.Module):
    """Residual block for the encoder and decoder."""
    
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3, 
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # Apply spectral normalization if specified
        if use_spectral_norm:
            self.conv1 = SpectralNorm.apply(self.conv1)
            self.conv2 = SpectralNorm.apply(self.conv2)
        
        # Normalizations and activations
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor [batch, channels, height, width]
        """
        residual = x
        
        # First convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Add residual connection
        x = x + residual
        x = self.activation(x)
        
        return x


class Encoder(nn.Module):
    """VAE encoder network that maps mel spectrograms to latent distributions."""
    
    def __init__(
        self, 
        in_channels: int = 1, 
        hidden_channels: List[int] = [16, 32, 64], 
        latent_dim: int = 32,
        use_residual: bool = True,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Initial projection
        self.init_conv = nn.Conv2d(
            in_channels, 
            hidden_channels[0], 
            kernel_size=3, 
            padding=1
        )
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            # Add residual block if specified
            if use_residual:
                self.down_blocks.append(
                    ResidualBlock(
                        hidden_channels[i],
                        use_spectral_norm=use_spectral_norm
                    )
                )
            
            # Add downsampling convolution
            down_conv = nn.Conv2d(
                hidden_channels[i],
                hidden_channels[i+1],
                kernel_size=4,
                stride=2,
                padding=1
            )
            
            # Apply spectral normalization if specified
            if use_spectral_norm:
                down_conv = SpectralNorm.apply(down_conv)
            
            self.down_blocks.append(down_conv)
            self.down_blocks.append(nn.BatchNorm2d(hidden_channels[i+1]))
            self.down_blocks.append(nn.LeakyReLU(0.2))
        
        # Final residual block
        if use_residual:
            self.down_blocks.append(
                ResidualBlock(
                    hidden_channels[-1],
                    use_spectral_norm=use_spectral_norm
                )
            )
        
        # To latent parameters (mean and logvar)
        self.to_latent = nn.Conv2d(
            hidden_channels[-1],
            latent_dim * 2,  # Mean and logvar
            kernel_size=3,
            padding=1
        )
        
        # Apply spectral normalization if specified
        if use_spectral_norm:
            self.init_conv = SpectralNorm.apply(self.init_conv)
            self.to_latent = SpectralNorm.apply(self.to_latent)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            
        Returns:
            Tuple of (mean, logvar) each of shape [batch, latent_dim, height/2^n, width/2^n]
            where n is the number of downsampling layers
        """
        # Initial projection
        x = self.init_conv(x)
        
        # Apply down blocks
        for block in self.down_blocks:
            x = block(x)
        
        # Get latent parameters
        latent_params = self.to_latent(x)
        
        # Split into mean and logvar
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        
        return mu, logvar


class Decoder(nn.Module):
    """VAE decoder network that maps latent vectors back to mel spectrograms."""
    
    def __init__(
        self, 
        latent_dim: int = 32, 
        hidden_channels: List[int] = [64, 32, 16], 
        out_channels: int = 1,
        use_residual: bool = True,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Initial projection from latent
        self.init_conv = nn.Conv2d(
            latent_dim, 
            hidden_channels[0], 
            kernel_size=3, 
            padding=1
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            # Add residual block if specified
            if use_residual:
                self.up_blocks.append(
                    ResidualBlock(
                        hidden_channels[i],
                        use_spectral_norm=use_spectral_norm
                    )
                )
            
            # Add upsampling transposed convolution
            up_conv = nn.ConvTranspose2d(
                hidden_channels[i],
                hidden_channels[i+1],
                kernel_size=4,
                stride=2,
                padding=1
            )
            
            # Apply spectral normalization if specified
            if use_spectral_norm:
                up_conv = SpectralNorm.apply(up_conv)
            
            self.up_blocks.append(up_conv)
            self.up_blocks.append(nn.BatchNorm2d(hidden_channels[i+1]))
            self.up_blocks.append(nn.LeakyReLU(0.2))
        
        # Final residual block
        if use_residual:
            self.up_blocks.append(
                ResidualBlock(
                    hidden_channels[-1],
                    use_spectral_norm=use_spectral_norm
                )
            )
        
        # Final convolution to output
        self.to_output = nn.Conv2d(
            hidden_channels[-1],
            out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Apply spectral normalization if specified
        if use_spectral_norm:
            self.init_conv = SpectralNorm.apply(self.init_conv)
            self.to_output = SpectralNorm.apply(self.to_output)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output
        
        Args:
            x: Latent tensor [batch, latent_dim, height, width]
            
        Returns:
            Output tensor [batch, out_channels, height*2^n, width*2^n]
            where n is the number of upsampling layers
        """
        # Initial projection
        x = self.init_conv(x)
        
        # Apply up blocks
        for block in self.up_blocks:
            x = block(x)
        
        # Final convolution to output
        x = self.to_output(x)
        
        return torch.tanh(x)  # Use tanh for bounded output


class VAE(BaseModule):
    """
    Variational Autoencoder for mel spectrograms with KL annealing support.
    """
    
    def _build_model(self):
        """Build the VAE architecture."""
        config = self.config
        
        # Use configuration options
        in_channels = 1  # Mel spectrogram has 1 channel
        latent_dim = config["latent_dim"]
        encoder_channels = config["encoder_channels"]
        decoder_channels = config["decoder_channels"]
        use_residual = config.get("use_residual_blocks", True)
        use_spectral_norm = config.get("use_spectral_norm", True)
        
        # Initialize KL weight
        self.kl_weight = config["kl_weight"]["initial"]
        self.initial_kl_weight = config["kl_weight"]["initial"]
        self.target_kl_weight = config["kl_weight"]["target"]
        self.kl_anneal_steps = config["kl_weight"]["anneal_steps"]
        
        # Create encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=encoder_channels,
            latent_dim=latent_dim,
            use_residual=use_residual,
            use_spectral_norm=use_spectral_norm
        )
        
        # Create decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_channels=decoder_channels,
            out_channels=in_channels,
            use_residual=use_residual,
            use_spectral_norm=use_spectral_norm
        )
        
        # Initialize latent channels for dimension tracking
        self.latent_channels = latent_dim
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent representation
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Tuple of (z, mu, logvar) where:
                z: Sampled latent vector [batch, latent_dim, height/2^n, width/2^n]
                mu: Mean of latent distribution [batch, latent_dim, height/2^n, width/2^n]
                logvar: Log variance of latent distribution [batch, latent_dim, height/2^n, width/2^n]
        """
        # Get distribution parameters
        mu, logvar = self.encoder(x)
        
        # Sample from distribution
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output
        
        Args:
            z: Latent tensor [batch, latent_dim, height, width]
            
        Returns:
            Reconstructed output [batch, channels, height*2^n, width*2^n]
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def update_kl_weight(self, global_step: int):
        """
        Update KL weight based on annealing schedule
        
        Args:
            global_step: Current training step
        """
        if global_step > self.kl_anneal_steps:
            self.kl_weight = self.target_kl_weight
        else:
            self.kl_weight = self.initial_kl_weight + (
                self.target_kl_weight - self.initial_kl_weight
            ) * (global_step / self.kl_anneal_steps)
    
    def forward(
        self, 
        x: torch.Tensor, 
        global_step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through VAE
        
        Args:
            x: Input tensor [batch, channels, height, width]
            global_step: Current training step (for KL annealing)
            
        Returns:
            Tuple of (x_recon, z, loss, recon_loss, kl_loss) where:
                x_recon: Reconstructed output [batch, channels, height, width]
                z: Sampled latent vector [batch, latent_dim, height/2^n, width/2^n]
                loss: Total loss (weighted sum of recon_loss and kl_loss)
                recon_loss: Reconstruction loss
                kl_loss: KL divergence loss
        """
        # Update KL weight if global step is provided
        if global_step is not None:
            self.update_kl_weight(global_step)
            
        # Encode
        z, mu, logvar = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        # Ensure x_recon has the same shape as x
        if x_recon.shape != x.shape:
            x_recon = F.interpolate(
                x_recon, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / torch.numel(mu)  # Normalize by total number of elements
        
        # Compute total loss
        loss = recon_loss + self.kl_weight * kl_loss
        
        return x_recon, z, loss, recon_loss, kl_loss
        
    def get_latent_shape(
        self, 
        mel_height: int, 
        mel_width: int
    ) -> Tuple[int, int]:
        """
        Calculate latent space dimensions based on input dimensions
        
        Args:
            mel_height: Height of input mel spectrogram
            mel_width: Width of input mel spectrogram
            
        Returns:
            Tuple of (latent_height, latent_width)
        """
        # Calculate downsampling factor based on encoder architecture
        downsample_factor = 2 ** (len(self.config["encoder_channels"]) - 1)
        
        # Calculate latent dimensions
        latent_height = mel_height // downsample_factor
        latent_width = mel_width // downsample_factor
        
        return latent_height, latent_width