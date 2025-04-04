import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: Query tensor [batch, seq_len, query_dim]
            context: Context tensor [batch, context_len, context_dim]
            
        Returns:
            Attention output [batch, seq_len, query_dim]
        """
        batch_size, seq_len, _ = x.shape
        h = self.heads
        
        # Project to queries, keys, values
        q = self.to_q(x)                  # [batch, seq_len, inner_dim]
        k = self.to_k(context)            # [batch, context_len, inner_dim]
        v = self.to_v(context)            # [batch, context_len, inner_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, h, -1).permute(0, 2, 1, 3)      # [batch, heads, seq_len, dim_head]
        k = k.view(batch_size, -1, h, q.size(-1)).permute(0, 2, 1, 3)   # [batch, heads, context_len, dim_head]
        v = v.view(batch_size, -1, h, q.size(-1)).permute(0, 2, 1, 3)   # [batch, heads, context_len, dim_head]

        # Calculate attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [batch, heads, seq_len, context_len]
        attn = F.softmax(sim, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)   # [batch, heads, seq_len, dim_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # [batch, seq_len, inner_dim]
        
        return self.to_out(out)


class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, time_dim=None, groups=8):
        super().__init__()
        self.time_mlp = None
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, dim_out)
            )
        
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        Args:
            x: Input tensor [batch, channels, height, width]
            time_emb: Time embedding [batch, time_dim]
            
        Returns:
            Output tensor [batch, dim_out, height, width]
        """
        h = self.block1(x)
        
        if self.time_mlp and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb.reshape(time_emb.shape[0], -1, 1, 1)
            
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, 
                in_channels=32, 
                model_channels=128, 
                out_channels=32,
                time_dim=256,
                context_dim=None,
                attention_levels=[1, 2],
                num_groups=8):
        super().__init__()
        
        self.time_dim = time_dim
        self.context_dim = context_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim // 4),
            nn.Linear(time_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList([])
        channels = [model_channels]
        now_channels = model_channels
        
        for i in range(3):  # 3 levels of downsampling
            channel_mult = 2 ** i
            out_channels = model_channels * channel_mult
            
            self.downs.append(ConvBlock(now_channels, out_channels, time_dim, num_groups))
            
            # Add cross-attention if needed
            if context_dim is not None and i in attention_levels:
                self.downs.append(CrossAttention(out_channels, context_dim))
            
            # Add downsampling
            self.downs.append(nn.Conv2d(out_channels, out_channels, 4, 2, 1))
            
            now_channels = out_channels
            channels.append(now_channels)
        
        # Middle block
        self.mid_block1 = ConvBlock(now_channels, now_channels, time_dim, num_groups)
        
        if context_dim is not None:
            self.mid_attn = CrossAttention(now_channels, context_dim)
            
        self.mid_block2 = ConvBlock(now_channels, now_channels, time_dim, num_groups)
        
        # Upsampling
        self.ups = nn.ModuleList([])
        for i in reversed(range(3)):  # 3 levels of upsampling
            channel_mult = 2 ** i
            out_channels = model_channels * channel_mult
            
            # Upsample
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(now_channels, out_channels, 3, padding=1)
            ))
            
            # Combine with skip connection
            self.ups.append(ConvBlock(out_channels * 2, out_channels, time_dim, num_groups))
            
            # Add cross-attention if needed
            if context_dim is not None and i in attention_levels:
                self.ups.append(CrossAttention(out_channels, context_dim))
            
            now_channels = out_channels
        
        # Final block
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t, context=None):
        """
        Forward pass through UNet
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            t: Timestep tensor [batch]
            context: Optional conditioning [batch, seq_len, context_dim]
            
        Returns:
            Output tensor [batch, out_channels, height, width]
        """
        # Time embedding
        t = t.to(torch.float32)
        t = self.time_mlp(t)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Store skip connections
        skips = [h]
        
        # Downsample
        for i, block in enumerate(self.downs):
            if isinstance(block, ConvBlock):
                h = block(h, t)
            elif isinstance(block, CrossAttention) and context is not None:
                # Reshape for cross-attention: [b,c,h,w] -> [b,h*w,c]
                b, c, height, width = h.shape
                h_flat = h.reshape(b, c, -1).permute(0, 2, 1)  # [b, h*w, c]
                h_flat = block(h_flat, context)  # [b, h*w, c]
                h = h_flat.permute(0, 2, 1).reshape(b, c, height, width)  # [b, c, h, w]
            else:
                h = block(h)
                
            if i % 3 == 2:  # After each downsampling operation
                skips.append(h)
        
        # Middle
        h = self.mid_block1(h, t)
        
        if hasattr(self, 'mid_attn') and context is not None:
            # Reshape for cross-attention
            b, c, height, width = h.shape
            h_flat = h.reshape(b, c, -1).permute(0, 2, 1)  # [b, h*w, c]
            h_flat = self.mid_attn(h_flat, context)  # [b, h*w, c]
            h = h_flat.permute(0, 2, 1).reshape(b, c, height, width)  # [b, c, h, w]
            
        h = self.mid_block2(h, t)
        
        # Upsample
        for i, block in enumerate(self.ups):
            if i % 3 == 0:  # Before each upsampling operation
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                
            if isinstance(block, ConvBlock):
                h = block(h, t)
            elif isinstance(block, CrossAttention) and context is not None:
                # Reshape for cross-attention
                b, c, height, width = h.shape
                h_flat = h.reshape(b, c, -1).permute(0, 2, 1)  # [b, h*w, c]
                h_flat = block(h_flat, context)  # [b, h*w, c]
                h = h_flat.permute(0, 2, 1).reshape(b, c, height, width)  # [b, c, h, w]
            else:
                h = block(h)
        
        # Final
        return self.final_conv(h)


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Diffusion parameters
        diffusion_config = config['model']['diffusion']
        self.num_timesteps = diffusion_config['diffusion_steps']
        self.beta_start = diffusion_config['beta_start']
        self.beta_end = diffusion_config['beta_end']
        
        # Register buffer for betas, alphas etc.
        if diffusion_config['beta_schedule'] == 'linear':
            self.register_buffer('betas', torch.linspace(self.beta_start, self.beta_end, self.num_timesteps))
        elif diffusion_config['beta_schedule'] == 'cosine':
            self.register_buffer('betas', self._cosine_beta_schedule(self.num_timesteps))
        
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / self.alphas))
        
        # Calculate posterior variance (important for sampling)
        posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))
        
        # Model dimensions
        self.latent_channels = config['model']['vae']['latent_channels']
        self.context_dim = config['model']['conditioning']['condition_channels']
        
        # Noise prediction model
        self.model = UNet(
            in_channels=self.latent_channels,
            model_channels=self.latent_channels * 2,
            out_channels=self.latent_channels,
            time_dim=256,
            context_dim=self.context_dim
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample from q(x_t | x_0)
        
        Args:
            x_start: Initial sample [batch, channels, height, width]
            t: Timestep [batch]
            noise: Optional noise to add [batch, channels, height, width]
            
        Returns:
            Noisy sample at timestep t [batch, channels, height, width]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract_into_tensor(self, a, t, broadcast_shape):
        """
        Extract values from a 1-D torch tensor for a batch of indices.
        """
        t = t.long()
        out = a.gather(-1, t).reshape(-1, 1, 1, 1)
        return out.expand(broadcast_shape)
    
    def p_losses(self, x_start, t, context=None, noise=None):
        """
        Training loss calculation
        
        Args:
            x_start: Initial sample [batch, channels, height, width]
            t: Timestep [batch]
            context: Conditioning information [batch, seq_len, channels]
            noise: Optional noise to add [batch, channels, height, width]
            
        Returns:
            Loss scalar
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, context)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def p_sample(self, x, t, context=None):
        """
        Reverse diffusion: sample from p(x_{t-1} | x_t)
        
        Args:
            x: Current sample [batch, channels, height, width]
            t: Current timestep [batch]
            context: Conditioning information [batch, seq_len, channels]
            
        Returns:
            Sample at timestep t-1 [batch, channels, height, width]
        """
        model_output = self.model(x, t, context)
        
        # Get posterior parameters
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        posterior_mean_coef1 = self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2 = self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape)
        
        # Calculate posterior mean
        posterior_mean = posterior_mean_coef1 * x + posterior_mean_coef2 * model_output
        
        # Sample
        noise = torch.randn_like(x)
        nonzero_mask = (t > 0).float().reshape(-1, 1, 1, 1)
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance_clipped) * noise
    
    def p_sample_loop(self, shape, context=None):
        """
        Full sampling loop for inference
        
        Args:
            shape: Output shape [batch, channels, height, width]
            context: Conditioning information [batch, seq_len, channels]
            
        Returns:
            Generated sample [batch, channels, height, width]
        """
        device = self.betas.device
        b = shape[0]
        
        # Start with random noise
        img = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, context)
            
        return img
    
    def forward(self, x, context=None):
        """
        Forward pass: calculate loss for training
        
        Args:
            x: Input tensor [batch, channels, height, width] 
            context: Conditioning information [batch, seq_len, channels]
            
        Returns:
            Loss scalar
        """
        b, c, h, w = x.shape
        device = x.device
        
        # Sample random timesteps for each image in the batch
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        return self.p_losses(x, t, context)