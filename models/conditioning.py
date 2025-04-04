import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple, Optional, List
from models.base_model import BaseModule


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based architectures."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (won't be trained)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [batch, sequence_length, features]
            
        Returns:
            Output with positional encoding added [batch, sequence_length, features]
        """
        return self.dropout(x + self.pe[:, :x.size(1)])


class ConvBlock(nn.Module):
    """1D Convolutional block with normalization and activation."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        dilation: int = 1, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2 * dilation,
            dilation=dilation
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional block
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            Processed tensor [batch, channels, time]
        """
        x = self.conv(x)
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [batch, channels, time]
        return x


class ConditionalResidualBlock(nn.Module):
    """Residual block with optional conditioning."""
    
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3, 
        dilation: int = 1,
        condition_channels: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.has_conditioning = condition_channels is not None
        
        # First conv block
        self.conv1 = ConvBlock(
            channels, 
            channels, 
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout
        )
        
        # Optional conditioning projection
        if self.has_conditioning:
            self.cond_proj = nn.Linear(condition_channels, channels * 2)
        
        # Second conv block
        self.conv2 = ConvBlock(
            channels, 
            channels, 
            kernel_size=kernel_size,
            dilation=dilation * 2,  # Increasing dilation
            dropout=dropout
        )
        
        # Residual connection
        self.residual = nn.Identity()
        
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply residual block with optional conditioning
        
        Args:
            x: Input tensor [batch, channels, time]
            condition: Optional conditioning tensor [batch, time, condition_channels]
            
        Returns:
            Processed tensor [batch, channels, time]
        """
        residual = self.residual(x)
        
        # First convolution
        x = self.conv1(x)
        
        # Apply conditioning if available
        if self.has_conditioning and condition is not None:
            # Ensure condition has the right shape
            if condition.dim() == 3:  # [batch, time, channels]
                condition = condition.transpose(1, 2)  # [batch, channels, time]
                
            # Project condition to scale and shift
            cond = self.cond_proj(condition.transpose(1, 2))  # [batch, time, channels*2]
            cond = cond.transpose(1, 2)  # [batch, channels*2, time]
            
            # Split into scale and shift
            scale, shift = torch.chunk(cond, 2, dim=1)
            
            # Apply scale and shift (FiLM conditioning)
            x = x * (1 + scale) + shift
        
        # Second convolution
        x = self.conv2(x)
        
        # Add residual connection
        x = x + residual
        
        return x


class FeatureTokenizer(nn.Module):
    """Convert raw features into embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Get dimensions from config
        self.f0_embedding_dim = config["f0_embedding_dim"]
        self.phone_embedding_dim = config["phone_embedding_dim"]
        self.midi_embedding_dim = config["midi_embedding_dim"]
        self.duration_embedding_dim = config["duration_embedding_dim"]
        
        # F0 processing (continuous value)
        self.f0_encoder = nn.Sequential(
            nn.Linear(1, self.f0_embedding_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.f0_embedding_dim // 2),
            nn.Linear(self.f0_embedding_dim // 2, self.f0_embedding_dim),
            nn.LayerNorm(self.f0_embedding_dim)
        )
        
        # Phone embedding (categorical)
        # Assuming a vocabulary size of 100 (adjust as needed)
        self.phone_embedding = nn.Embedding(100, self.phone_embedding_dim)
        
        # MIDI note embedding (categorical)
        # 128 MIDI notes (0-127) + 1 for padding/unknown
        self.midi_embedding = nn.Embedding(129, self.midi_embedding_dim)
        
        # Duration processing (continuous value)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, self.duration_embedding_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.duration_embedding_dim // 2),
            nn.Linear(self.duration_embedding_dim // 2, self.duration_embedding_dim),
            nn.LayerNorm(self.duration_embedding_dim)
        )
        
    def forward(
        self, 
        f0: torch.Tensor, 
        phones: torch.Tensor, 
        durations: torch.Tensor, 
        midi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize raw features into embeddings
        
        Args:
            f0: F0 values [batch, time]
            phones: Phone IDs [batch, time]
            durations: Duration values [batch, time]
            midi: MIDI note IDs [batch, time]
            
        Returns:
            Tuple of embedded features, each [batch, time, dim]
        """
        # Process F0 (continuous)
        f0 = f0.unsqueeze(-1)  # [batch, time, 1]
        f0_emb = self.f0_encoder(f0)  # [batch, time, f0_dim]
        
        # Process phones (categorical)
        phone_emb = self.phone_embedding(phones)  # [batch, time, phone_dim]
        
        # Process durations (continuous)
        dur = durations.unsqueeze(-1)  # [batch, time, 1]
        dur_emb = self.duration_encoder(dur)  # [batch, time, dur_dim]
        
        # Process MIDI (categorical)
        midi_emb = self.midi_embedding(midi)  # [batch, time, midi_dim]
        
        return f0_emb, phone_emb, dur_emb, midi_emb


class ConditioningEncoder(BaseModule):
    """
    Encodes and combines all conditioning information (F0, phonemes, duration, MIDI)
    into a unified conditioning representation.
    """
    
    def _build_model(self):
        """Build the conditioning encoder architecture."""
        config = self.config
        
        # Create feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(config)
        
        # Calculate total embedding dimension
        total_embedding_dim = (
            config["f0_embedding_dim"] + 
            config["phone_embedding_dim"] + 
            config["midi_embedding_dim"] + 
            config["duration_embedding_dim"]
        )
        
        # Initial projection from combined embeddings to hidden dimension
        self.initial_projection = nn.Linear(
            total_embedding_dim, 
            config["hidden_channels"]
        )
        
        # Optional positional encoding
        self.use_pos_encoding = config.get("use_pos_encoding", True)
        if self.use_pos_encoding:
            self.pos_encoding = PositionalEncoding(
                config["hidden_channels"], 
                dropout=config["dropout"]
            )
        
        # Residual blocks for feature extraction
        self.res_blocks = nn.ModuleList([
            ConditionalResidualBlock(
                channels=config["hidden_channels"],
                kernel_size=config["kernel_size"],
                dilation=2**i,  # Exponential dilation
                dropout=config["dropout"]
            )
            for i in range(config["n_layers"])
        ])
        
        # Final projection
        self.final_projection = nn.Conv1d(
            config["hidden_channels"],
            config["hidden_channels"],
            kernel_size=1
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(config["hidden_channels"])
        
    def forward(
        self, 
        f0: torch.Tensor, 
        phones: torch.Tensor, 
        durations: torch.Tensor, 
        midi: torch.Tensor
    ) -> torch.Tensor:
        """
        Process and combine all conditioning inputs
        
        Args:
            f0: Fundamental frequency contour [batch, time_steps]
            phones: Phone labels [batch, time_steps]
            durations: Phone durations [batch, time_steps] 
            midi: MIDI note labels [batch, time_steps]
            
        Returns:
            Combined conditioning embeddings [batch, time_steps, hidden_channels]
        """
        # Get feature embeddings
        f0_emb, phone_emb, dur_emb, midi_emb = self.feature_tokenizer(
            f0, phones, durations, midi
        )
        
        # Combine embeddings along feature dimension
        combined = torch.cat([f0_emb, phone_emb, dur_emb, midi_emb], dim=-1)
        
        # Initial projection to hidden dimension
        x = self.initial_projection(combined)  # [batch, time, hidden]
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            x = self.pos_encoding(x)
        
        # Transpose for 1D convolution operations [batch, hidden, time]
        x = x.transpose(1, 2)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Final projection
        x = self.final_projection(x)
        
        # Back to [batch, time, hidden]
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class FeatureAligner(BaseModule):
    """
    Aligns and resamples condition features to match the target temporal resolution,
    such as aligning to the latent space time dimension.
    """
    
    def _build_model(self):
        """Build the feature aligner architecture."""
        config = self.config
        
        # Input and output dimensions
        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.target_length = config.get("target_length", None)
        
        # 1D Convolution for feature transformation
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Normalization and activation
        self.norm = nn.LayerNorm(self.out_channels)
        self.activation = nn.GELU()
        
        # Add positional encoding for better temporal awareness
        self.pos_encoding = PositionalEncoding(
            self.out_channels, 
            dropout=config.get("dropout", 0.1)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Align features to target temporal resolution
        
        Args:
            x: Input features [batch, time, channels]
            target_length: Target sequence length (overrides config)
            
        Returns:
            Aligned features [batch, target_length, out_channels]
        """
        # Use provided target length or fall back to config
        target_length = target_length or self.target_length
        
        if target_length is None:
            raise ValueError("Target length must be provided either in config or as argument")
        
        # Convert to [batch, channels, time] for Conv1D
        x = x.transpose(1, 2)  # [batch, channels, time]
        
        # Apply feature transformation
        x = self.conv(x)
        x = F.gelu(x)
        
        # Resample to target length using interpolation
        # Check if we need to interpolate
        if x.shape[2] != target_length:
            x = F.interpolate(
                x, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )
        
        # Convert back to [batch, time, channels]
        x = x.transpose(1, 2)  # [batch, target_length, channels]
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        return x