import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union, Callable
from models.base_model import BaseModule


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate position embeddings for timestep
        
        Args:
            time: Timestep tensor [batch]
            
        Returns:
            Embeddings tensor [batch, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """Residual block with conditioning and attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        up: bool = False,
        down: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up = up
        self.down = down
        
        # Normalization and activation
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        
        # Up/down sampling if needed
        if up:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time projection for conditioning
        if time_channels > 0:
            self.time_proj = nn.Linear(time_channels, out_channels * 2 if use_scale_shift_norm else out_channels)
        
        # Second normalization and activation
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection if in_channels != out_channels
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply residual block with conditioning
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            time_emb: Time embedding [batch, time_channels]
            
        Returns:
            Output tensor [batch, out_channels, height', width']
            where height' and width' depend on up/down sampling
        """
        # Save input for skip connection
        residual = x
        
        # First normalization and activation
        h = self.norm1(x)
        h = self.act1(h)
        
        # Up/down sample if needed
        if self.up:
            h = self.upsample(h)
            residual = self.upsample(residual)
        elif self.down:
            h = self.downsample(h)
            residual = self.downsample(residual)
        
        # First convolution
        h = self.conv1(h)
        
        # Apply time conditioning
        if time_emb is not None:
            assert self.time_channels > 0
            
            # Project time embedding
            time_emb = self.time_proj(time_emb)
            
            if self.use_scale_shift_norm:
                # Split into scale and shift for normalization
                scale, shift = torch.chunk(time_emb, 2, dim=1)
                
                # Reshape for broadcasting
                scale = scale.view(-1, self.out_channels, 1, 1)
                shift = shift.view(-1, self.out_channels, 1, 1)
                
                # Apply scale and shift after normalization
                h = self.norm2(h) * (1 + scale) + shift
                h = self.act2(h)
            else:
                # Add time embedding directly
                time_emb = time_emb.view(-1, self.out_channels, 1, 1)
                h = self.norm2(h) + time_emb
                h = self.act2(h)
        else:
            # Regular normalization and activation
            h = self.norm2(h)
            h = self.act2(h)
        
        # Dropout
        h = self.dropout(h)
        
        # Second convolution
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_connection(residual)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial dependencies."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Normalization
        self.norm = nn.GroupNorm(32, channels)
        
        # QKV projection
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Scale factor for attention
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor [batch, channels, height, width]
        """
        # Save input for skip connection
        residual = x
        
        # Normalization
        x = self.norm(x)
        
        # Get shape
        batch, channels, height, width = x.shape
        
        # Project to QKV
        qkv = self.qkv(x)
        
        # Reshape and split for multi-head attention
        qkv = qkv.reshape(batch, 3 * channels, height * width)
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention scores
        attn = torch.einsum("bhnc,bhnd->bhcd", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum("bhcd,bhnd->bhnc", attn, v)
        
        # Reshape back
        out = out.reshape(batch, channels, height, width)
        
        # Output projection
        out = self.proj(out)
        
        # Skip connection
        return out + residual


class CrossAttentionBlock(nn.Module):
    """Cross-attention for conditioning information."""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Normalization
        self.norm = nn.GroupNorm(32, query_dim)
        
        # Query, key, and value projections
        self.to_q = nn.Conv2d(query_dim, query_dim, kernel_size=1)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        
        # Output projection
        self.proj = nn.Conv2d(query_dim, query_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention
        
        Args:
            x: Query tensor [batch, channels, height, width]
            context: Context tensor [batch, seq_len, context_dim]
            
        Returns:
            Output tensor [batch, channels, height, width]
        """
        # Save input for skip connection
        residual = x
        
        # Normalization
        x = self.norm(x)
        
        # Get shape
        batch, channels, height, width = x.shape
        
        # Project query
        q = self.to_q(x)
        q = q.reshape(batch, self.num_heads, self.head_dim, height * width)
        q = q.permute(0, 1, 3, 2)  # [batch, num_heads, hw, head_dim]
        
        # Project key and value
        k = self.to_k(context)
        v = self.to_v(context)
        
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        
        # Apply attention to values
        out = torch.einsum("bijh,bjhd->bihd", attn, v)
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        
        # Output projection
        out = self.proj(out)
        
        # Skip connection
        return out + residual


class UNet(nn.Module):
    """UNet architecture for diffusion model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Get configuration
        unet_config = config["unet"]
        
        # Model dimensions
        self.in_channels = config.get("in_channels", 1)
        self.base_channels = unet_config["base_channels"]
        self.channel_multipliers = unet_config["channel_multipliers"]
        self.num_res_blocks = unet_config["num_res_blocks"]
        self.attention_levels = unet_config["attention_levels"]
        self.num_heads = unet_config["num_heads"]
        self.use_scale_shift_norm = unet_config["use_scale_shift_norm"]
        self.dropout = unet_config["dropout"]
        
        # Conditioning
        self.context_dim = config.get("context_dim", None)
        self.time_dim = self.base_channels * 4
        
        # Build time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.base_channels),
            nn.Linear(self.base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        # Input projection
        self.input_projection = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        
        # Track input/output channels for each level
        input_channels = self.base_channels
        channels_list = [input_channels]
        
        # For each level in the UNet
        for i, mult in enumerate(self.channel_multipliers):
            output_channels = self.base_channels * mult
            
            # For each residual block at this level
            for j in range(self.num_res_blocks):
                # Add residual block
                self.down_blocks.append(
                    ResnetBlock(
                        input_channels,
                        output_channels,
                        time_channels=self.time_dim,
                        dropout=self.dropout,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        down=(j == 0 and i > 0)  # Downsample on first block except level 0
                    )
                )
                
                input_channels = output_channels
                channels_list.append(input_channels)
                
                # Add attention if needed
                if i in self.attention_levels:
                    # Self-attention
                    self.down_blocks.append(
                        AttentionBlock(
                            input_channels,
                            num_heads=self.num_heads
                        )
                    )
                    
                    # Cross-attention if context is provided
                    if self.context_dim is not None:
                        self.down_blocks.append(
                            CrossAttentionBlock(
                                input_channels,
                                self.context_dim,
                                num_heads=self.num_heads
                            )
                        )
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResnetBlock(
                input_channels,
                input_channels,
                time_channels=self.time_dim,
                dropout=self.dropout,
                use_scale_shift_norm=self.use_scale_shift_norm
            ),
            AttentionBlock(
                input_channels,
                num_heads=self.num_heads
            )
        ])
        
        # Cross-attention in middle if context is provided
        if self.context_dim is not None:
            self.middle_blocks.append(
                CrossAttentionBlock(
                    input_channels,
                    self.context_dim,
                    num_heads=self.num_heads
                )
            )
            
        self.middle_blocks.append(
            ResnetBlock(
                input_channels,
                input_channels,
                time_channels=self.time_dim,
                dropout=self.dropout,
                use_scale_shift_norm=self.use_scale_shift_norm
            )
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        
        # For each level in the UNet (reversed)
        for i, mult in reversed(list(enumerate(self.channel_multipliers))):
            output_channels = self.base_channels * mult
            
            # For each residual block at this level
            for j in range(self.num_res_blocks + 1):
                # Get skip connection channels
                skip_channels = channels_list.pop()
                
                # Add residual block with skip connection
                self.up_blocks.append(
                    ResnetBlock(
                        skip_channels + input_channels,
                        output_channels,
                        time_channels=self.time_dim,
                        dropout=self.dropout,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        up=(j == self.num_res_blocks and i > 0)  # Upsample on last block except level 0
                    )
                )
                
                input_channels = output_channels
                
                # Add attention if needed
                if i in self.attention_levels:
                    # Self-attention
                    self.up_blocks.append(
                        AttentionBlock(
                            input_channels,
                            num_heads=self.num_heads
                        )
                    )
                    
                    # Cross-attention if context is provided
                    if self.context_dim is not None:
                        self.up_blocks.append(
                            CrossAttentionBlock(
                                input_channels,
                                self.context_dim,
                                num_heads=self.num_heads
                            )
                        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.GroupNorm(32, input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, self.in_channels, kernel_size=3, padding=1)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        time: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through UNet
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
            time: Timestep tensor [batch]
            context: Optional conditioning [batch, seq_len, context_dim]
            
        Returns:
            Output tensor [batch, in_channels, height, width]
        """
        # Time embedding
        time_emb = self.time_embed(time)
        
        # Initial projection
        h = self.input_projection(x)
        
        # Store skip connections
        skips = [h]
        
        # Down blocks
        for i, block in enumerate(self.down_blocks):
            if isinstance(block, ResnetBlock):
                h = block(h, time_emb)
                skips.append(h)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            elif isinstance(block, CrossAttentionBlock) and context is not None:
                h = block(h, context)
        
        # Middle blocks
        for i, block in enumerate(self.middle_blocks):
            if isinstance(block, ResnetBlock):
                h = block(h, time_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            elif isinstance(block, CrossAttentionBlock) and context is not None:
                h = block(h, context)
        
        # Up blocks
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, ResnetBlock):
                # Get skip connection
                skip = skips.pop()
                
                # Align shapes if needed
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(
                        h, size=skip.shape[-2:], mode='bilinear', align_corners=True
                    )
                
                # Concatenate skip connection
                h = torch.cat([h, skip], dim=1)
                
                # Apply block
                h = block(h, time_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            elif isinstance(block, CrossAttentionBlock) and context is not None:
                h = block(h, context)
        
        # Output projection
        h = self.output_projection(h)
        
        return h


class DiffusionModel(BaseModule):
    """
    Diffusion model for latent space generation with conditioning.
    """
    
    def _build_model(self):
        """Build the diffusion model architecture."""
        config = self.config
        
        # Diffusion parameters
        self.diffusion_steps = config["diffusion_steps"]
        self.beta_start = config["beta_start"]
        self.beta_end = config["beta_end"]
        self.diffusion_schedule = config["diffusion_schedule"]
        
        # Model parameters
        self.prediction_type = config.get("prediction_type", "epsilon")
        assert self.prediction_type in ["epsilon", "x0", "v"], f"Unknown prediction type: {self.prediction_type}"
        
        # Conditioning parameters
        self.conditioning_mode = config.get("conditioning_mode", "cross_attention")
        
        # Register diffusion schedule parameters
        self._setup_diffusion_schedule()
        
        # Create UNet for noise prediction
        self.model = UNet({
            "in_channels": config.get("in_channels", 1),
            "context_dim": config.get("context_dim", None),
            "unet": config["unet"]
        })
        
    def _setup_diffusion_schedule(self):
        """Set up diffusion schedule parameters."""
        # Create beta schedule
        if self.diffusion_schedule == "linear":
            self.betas = torch.linspace(
                self.beta_start, 
                self.beta_end, 
                self.diffusion_steps
            )
        elif self.diffusion_schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = self.diffusion_steps + 1
            x = torch.linspace(0, self.diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / self.diffusion_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        elif self.diffusion_schedule == "quadratic":
            # Quadratic schedule
            self.betas = torch.linspace(
                self.beta_start ** 0.5, 
                self.beta_end ** 0.5, 
                self.diffusion_steps
            ) ** 2
        else:
            raise ValueError(f"Unknown diffusion schedule: {self.diffusion_schedule}")
        
        # Calculate related constants
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Constants for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.maximum(
            self.posterior_variance, torch.tensor(1e-20)
        ))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values at timesteps and reshape to match x_shape
        
        Args:
            a: Source tensor [diffusion_steps] to extract from
            t: Timestep indices [batch]
            x_shape: Target shape
            
        Returns:
            Extracted values broadcasted to x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu()).to(t.device)
        
        # Reshape to match target shape
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: sample from q(x_t | x_0)
        
        Args:
            x_start: Initial clean sample [batch, channels, height, width]
            t: Timestep indices [batch]
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of (noisy_sample, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Extract coefficients for this timestep
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Add noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise
        
        Args:
            x_t: Noisy sample at timestep t [batch, channels, height, width]
            t: Timestep indices [batch]
            noise: Predicted noise [batch, channels, height, width]
            
        Returns:
            Predicted x_0 [batch, channels, height, width]
        """
        # Extract coefficients for this timestep
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        # Predict x_0
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_start: Predicted x_0 [batch, channels, height, width]
            x_t: Current noisy sample x_t [batch, channels, height, width]
            t: Current timestep indices [batch]
            
        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        # Extract coefficients for this timestep
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        
        # Compute posterior mean
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        # Extract posterior variance and log variance
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t
    
    def p_mean_variance(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for the denoising step
        
        Args:
            x_t: Current noisy sample [batch, channels, height, width]
            t: Current timestep indices [batch]
            context: Conditioning information [batch, seq_len, context_dim]
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            
        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        # Predict noise or x_0 based on mode
        model_output = self.model(x_t, t, context)
        
        if self.prediction_type == "epsilon":
            # Model predicts noise
            predicted_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        elif self.prediction_type == "x0":
            # Model directly predicts x_0
            x_start = model_output
        else:  # v-prediction
            # Model predicts v = alpha * noise - sigma * x0
            v = model_output
            alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            
            # Derive x_0: (x_t - sigma_t * v) / alpha_t
            x_start = (x_t - sigma_t * v) / alpha_t
        
        # Clip x_0 prediction if requested
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        
        # Get posterior parameters
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, 
            x_t=x_t, 
            t=t
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        
        Args:
            x_t: Current noisy sample [batch, channels, height, width]
            t: Current timestep indices [batch]
            context: Conditioning information [batch, seq_len, context_dim]
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            
        Returns:
            Sample at timestep t-1 [batch, channels, height, width]
        """
        # Get posterior mean and variance
        posterior_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(
            x_t, t, context, clip_denoised
        )
        
        # Sample from posterior
        noise = torch.randn_like(x_t)
        
        # Don't add noise for t=0
        nonzero_mask = (t > 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
    
    def p_sample_loop(
        self, 
        shape: Tuple[int, ...], 
        context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Full sampling loop for inference
        
        Args:
            shape: Output shape [batch, channels, height, width]
            context: Conditioning information [batch, seq_len, context_dim]
            noise: Initial noise (if None, random noise is used)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            progress: Whether to show progress bar
            
        Returns:
            Generated sample [batch, channels, height, width]
        """
        device = self.betas.device
        batch_size = shape[0]
        
        # Start with random noise or provided noise
        img = noise if noise is not None else torch.randn(shape, device=device)
        
        # Setup progress range
        progress_fn = tqdm if progress else lambda x: x
        time_range = list(reversed(range(self.diffusion_steps)))
        
        # Iterative denoising
        for i in progress_fn(time_range):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, timesteps, context, clip_denoised)
            
        return img
    
    def p_losses(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Training loss calculation
        
        Args:
            x_start: Initial clean sample [batch, channels, height, width]
            t: Timestep indices [batch]
            context: Conditioning information [batch, seq_len, context_dim]
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Generate noise
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Forward diffusion to get x_t
        x_noisy, _ = self.q_sample(x_start, t, noise)
        
        # Predict noise or x_0
        model_output = self.model(x_noisy, t, context)
        
        # Calculate loss based on prediction type
        if self.prediction_type == "epsilon":
            # MSE to noise
            target = noise
        elif self.prediction_type == "x0":
            # MSE to x_0
            target = x_start
        else:  # v-prediction
            # v = alpha * noise - sigma * x0
            alpha_t = self._extract(self.alphas_cumprod, t, x_start.shape)
            sigma_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            target = alpha_t * noise - sigma_t * x_start
        
        # Calculate loss
        loss = F.mse_loss(model_output, target)
        
        # Return metrics
        metrics = {
            "loss": loss,
            "x_start_mean": x_start.mean().item(),
            "x_start_std": x_start.std().item(),
            "x_noisy_mean": x_noisy.mean().item(),
            "x_noisy_std": x_noisy.std().item(),
            "pred_mean": model_output.mean().item(),
            "pred_std": model_output.std().item(),
        }
        
        return loss, metrics
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        condition_dropout_prob: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: calculate loss for training with optional classifier-free guidance support
        
        Args:
            x: Input tensor [batch, channels, height, width]
            context: Conditioning information [batch, seq_len, context_dim]
            condition_dropout_prob: Probability of dropping conditioning for classifier-free guidance
            
        Returns:
            Tuple of (loss, metrics)
        """
        device = x.device
        b = x.shape[0]
        
        # Sample random timesteps for each item in batch
        t = torch.randint(0, self.diffusion_steps, (b,), device=device).long()
        
        # Apply conditioning dropout for classifier-free guidance
        if context is not None and condition_dropout_prob > 0.0:
            context_mask = torch.rand(b, device=device) >= condition_dropout_prob
            if not context_mask.all():
                # For items where conditioning is dropped, set context to None
                context_masked = []
                for i in range(b):
                    if context_mask[i]:
                        context_masked.append(context[i:i+1])
                    else:
                        # Create empty context with same shape
                        empty_context = torch.zeros_like(context[0:1])
                        context_masked.append(empty_context)
                
                context = torch.cat(context_masked, dim=0)
        
        # Calculate loss
        return self.p_losses(x, t, context)