import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Added missing import


class ConditioningEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get dimensions from config
        self.f0_embedding_dim = config['model']['conditioning']['f0_embedding_dim']
        self.phone_embedding_dim = config['model']['conditioning']['phone_embedding_dim']
        self.midi_embedding_dim = config['model']['conditioning']['midi_embedding_dim']
        self.duration_embedding_dim = config['model']['conditioning']['duration_embedding_dim']
        self.condition_channels = config['model']['conditioning']['condition_channels']
        
        # F0 processing (continuous value)
        self.f0_encoder = nn.Sequential(
            nn.Linear(1, self.f0_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.f0_embedding_dim // 2, self.f0_embedding_dim)
        )
        
        # Phone embedding (categorical)
        # Assuming a max of 100 phone categories - adjust as needed
        self.phone_embedding = nn.Embedding(100, self.phone_embedding_dim)
        
        # MIDI note embedding (categorical)
        # 128 MIDI notes (0-127)
        self.midi_embedding = nn.Embedding(128, self.midi_embedding_dim)
        
        # Duration processing (continuous value)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, self.duration_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.duration_embedding_dim // 2, self.duration_embedding_dim)
        )
        
        # Combine all embeddings
        total_embedding_dim = (
            self.f0_embedding_dim + 
            self.phone_embedding_dim + 
            self.midi_embedding_dim + 
            self.duration_embedding_dim
        )
        
        # Project to final conditioning dimension
        self.projection = nn.Sequential(
            nn.Linear(total_embedding_dim, self.condition_channels),
            nn.LayerNorm(self.condition_channels),
            nn.ReLU(),
            nn.Linear(self.condition_channels, self.condition_channels)
        )
    
    def forward(self, f0, phone, duration, midi):
        """
        Process and combine all conditioning inputs
        
        Args:
            f0: Fundamental frequency contour [batch, time_steps]
            phone: Phone labels [batch, time_steps]
            duration: Phone durations [batch, time_steps] 
            midi: MIDI note labels [batch, time_steps]
            
        Returns:
            Combined conditioning embeddings [batch, time_steps, condition_channels]
        """
        batch_size, time_steps = f0.shape
        
        # Process each time step independently to maintain batch structure
        f0_emb_list = []
        phone_emb_list = []
        midi_emb_list = []
        duration_emb_list = []
        
        for t in range(time_steps):
            # Process F0 (shape: [batch, 1])
            f0_t = f0[:, t:t+1]
            f0_emb = self.f0_encoder(f0_t)  # [batch, f0_dim]
            f0_emb_list.append(f0_emb)
            
            # Process phone labels (shape: [batch])
            phone_t = phone[:, t]
            phone_emb = self.phone_embedding(phone_t)  # [batch, phone_dim]
            phone_emb_list.append(phone_emb)
            
            # Process MIDI notes (shape: [batch])
            midi_t = midi[:, t]
            midi_emb = self.midi_embedding(midi_t)  # [batch, midi_dim]
            midi_emb_list.append(midi_emb)
            
            # Process durations (shape: [batch, 1])
            duration_t = duration[:, t:t+1]
            duration_emb = self.duration_encoder(duration_t)  # [batch, duration_dim]
            duration_emb_list.append(duration_emb)
        
        # Stack along time dimension
        f0_emb = torch.stack(f0_emb_list, dim=1)       # [batch, time, f0_dim]
        phone_emb = torch.stack(phone_emb_list, dim=1)  # [batch, time, phone_dim]
        midi_emb = torch.stack(midi_emb_list, dim=1)    # [batch, time, midi_dim]
        duration_emb = torch.stack(duration_emb_list, dim=1)  # [batch, time, duration_dim]
        
        # Concatenate along feature dimension
        combined = torch.cat([f0_emb, phone_emb, midi_emb, duration_emb], dim=2)  # [batch, time, total_dim]
        
        # Project to final dimension
        conditioning = self.projection(combined)  # [batch, time, condition_channels]
        
        return conditioning


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (won't be trained)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [batch, sequence_length, features]
            
        Returns:
            Output with positional encoding added [batch, sequence_length, features]
        """
        # Only add positional encoding up to the sequence length
        return x + self.pe[:, :x.size(1), :]


class FeatureAligner(nn.Module):
    """
    Aligns/resamples features to match latent space time dimension
    """
    def __init__(self, in_channels, out_channels, out_time_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_time_dim = out_time_dim
        
        # Use 1D convolution for time-aware feature extraction
        self.time_conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Add positional encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(out_channels)
        
    def forward(self, x, target_length=None):
        """
        Args:
            x: Features [batch, sequence_length, channels]
            target_length: Optional override for output time dimension
            
        Returns:
            Aligned features [batch, target_length, out_channels]
        """
        batch_size, seq_len, channels = x.shape
        
        # Use default target length if not specified
        if target_length is None:
            target_length = self.out_time_dim
            
        # Convert to [batch, channels, time] for Conv1D - proper transposition
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        
        # Apply temporal convolution
        x = self.time_conv(x)  # [batch, out_channels, seq_len]
        
        # Resample to target length using interpolation
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        # Now x is [batch, out_channels, target_length]
        
        # Convert back to [batch, time, channels]
        x = x.transpose(1, 2)  # [batch, target_length, out_channels]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        return x