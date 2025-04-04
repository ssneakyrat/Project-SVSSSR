import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, channels=[16, 32, 64], latent_channels=32):
        super().__init__()
        self.channels = channels
        
        # Initial projection
        self.proj = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Final convolution to latent space
        self.to_latent = nn.Conv2d(channels[-1], latent_channels*2, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial projection
        x = self.proj(x)
        
        # Down blocks
        for block in self.down_blocks:
            x = block(x)
        
        # To latent 
        x = self.to_latent(x)
        
        # Split into mean and log variance
        mu, log_var = torch.chunk(x, 2, dim=1)
        
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_channels=32, channels=[64, 32, 16], out_channels=1):
        super().__init__()
        self.channels = channels
        
        # Initial projection from latent space
        self.proj = nn.Conv2d(latent_channels, channels[0], kernel_size=3, padding=1)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Final convolution to output
        self.to_output = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial projection
        x = self.proj(x)
        
        # Up blocks
        for block in self.up_blocks:
            x = block(x)
        
        # To output 
        x = self.to_output(x)
        
        return x


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_channels = config['model']['vae']['latent_channels']
        self.kl_weight = config['model']['vae']['kl_weight']
        
        # Create encoder and decoder
        self.encoder = Encoder(
            in_channels=1,
            channels=config['model']['vae']['encoder_channels'],
            latent_channels=self.latent_channels
        )
        
        self.decoder = Decoder(
            latent_channels=self.latent_channels,
            channels=config['model']['vae']['decoder_channels'],
            out_channels=1
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # Encode
        z, mu, log_var = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        # Ensure x_recon matches x dimensions exactly before computing loss
        if x_recon.shape[2:] != x.shape[2:]:
            x_recon = F.interpolate(x_recon, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])
        kl_loss = torch.mean(kl_loss)
        
        # Compute reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        loss = recon_loss + self.kl_weight * kl_loss
        
        return x_recon, z, loss, recon_loss, kl_loss