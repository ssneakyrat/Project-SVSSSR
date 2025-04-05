import torch
import torch.nn as nn
import torch.nn.functional as F

class PhoneEncoder(nn.Module):
    """Encodes phoneme IDs into a continuous representation"""
    
    def __init__(self, vocab_size=50, embed_dim=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Phoneme IDs [batch, time]
        Returns:
            Phone embeddings [batch, time, embed_dim]
        """
        # [batch, time] -> [batch, time, embed_dim]
        x = self.embedding(x)
        
        # [batch, time, embed_dim] -> [batch, embed_dim, time]
        x = x.transpose(1, 2)
        
        # Apply 1D convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # [batch, embed_dim, time] -> [batch, time, embed_dim]
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        return x


class F0Encoder(nn.Module):
    """Encodes F0 contour into a continuous representation"""
    
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: F0 contour [batch, time, 1]
        Returns:
            F0 features [batch, time, hidden_dim]
        """
        # [batch, time, 1] -> [batch, 1, time]
        x = x.transpose(1, 2)
        
        # Apply 1D convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        # [batch, hidden_dim, time] -> [batch, time, hidden_dim]
        x = x.transpose(1, 2)
        
        return x


class DurationEncoder(nn.Module):
    """Encodes phone durations into a continuous representation"""
    
    def __init__(self, input_dim=1, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Phone durations [batch, time, 1]
        Returns:
            Duration features [batch, time, hidden_dim]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class MidiEncoder(nn.Module):
    """Encodes MIDI note IDs into a continuous representation"""
    
    def __init__(self, vocab_size=128, embed_dim=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: MIDI note IDs [batch, time]
        Returns:
            MIDI embeddings [batch, time, embed_dim]
        """
        # [batch, time] -> [batch, time, embed_dim]
        x = self.embedding(x)
        
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x