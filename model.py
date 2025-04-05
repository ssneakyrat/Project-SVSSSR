# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.plotting import plot_spectrograms_to_figure # Keep for validation plotting

# --- Helper Modules ---

class PositionalEncoding(nn.Module):
    """ Sinusoidal Positional Encoding """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, SeqLen, EmbDim)
        Returns:
            Tensor: Input tensor with added positional encoding.
        """
        # x.size(1) is the sequence length
        # self.pe shape: (1, max_len, d_model) -> select (1, SeqLen, d_model)
        return x + self.pe[:, :x.size(1)]

class FeedForwardNetwork(nn.Module):
    """ Position-wise Feed-Forward Network (using Conv1D as in Transformer TTS) """
    def __init__(self, hidden_dim, filter_size, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(filter_size, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, SeqLen, HiddenDim)
        Returns:
            Tensor: Output tensor of shape (Batch, SeqLen, HiddenDim)
        """
        # Transpose for Conv1D: (Batch, HiddenDim, SeqLen)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        # Transpose back: (Batch, SeqLen, HiddenDim)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    """ Multi-Head Self-Attention """
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.d_k = hidden_dim // n_heads
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (Tensor): Query tensor (Batch, SeqLen_Q, HiddenDim)
            key (Tensor): Key tensor (Batch, SeqLen_K, HiddenDim)
            value (Tensor): Value tensor (Batch, SeqLen_V, HiddenDim) (SeqLen_K == SeqLen_V)
            mask (Tensor, optional): Boolean mask (Batch, SeqLen_Q, SeqLen_K) or (Batch, 1, SeqLen_K). Defaults to None.
        Returns:
            Tensor: Output tensor (Batch, SeqLen_Q, HiddenDim)
        """
        batch_size = query.size(0)

        # Linear projections & split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (B, nH, SL_Q, d_k)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)   # (B, nH, SL_K, d_k)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (B, nH, SL_V, d_k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # (B, nH, SL_Q, SL_K)

        if mask is not None:
            # Ensure mask has compatible dimensions (Batch, nHeads, SeqLen_Q, SeqLen_K)
            # If mask is (B, SL_Q, SL_K), unsqueeze for heads dim
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # If mask is (B, 1, SL_K), it will broadcast across SeqLen_Q
            scores = scores.masked_fill(mask == 0, -1e9) # Fill with large negative value where mask is False

        attn_weights = torch.softmax(scores, dim=-1) # (B, nH, SL_Q, SL_K)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V) # (B, nH, SL_Q, d_k)

        # Concatenate heads and final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim) # (B, SL_Q, HiddenDim)
        output = self.fc(context)
        return output

class TransformerEncoderLayer(nn.Module):
    """ Single Transformer Encoder Layer """
    def __init__(self, hidden_dim, n_heads, filter_size, kernel_size, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, filter_size, kernel_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src (Tensor): Input tensor (Batch, SeqLen, HiddenDim)
            src_mask (Tensor, optional): Mask for self-attention (Batch, 1, SeqLen) or (Batch, SeqLen, SeqLen). Defaults to None.
        Returns:
            Tensor: Output tensor (Batch, SeqLen, HiddenDim)
        """
        # Self-Attention + Add & Norm
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        # Feed-Forward + Add & Norm
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_output))
        return src

class TransformerDecoderLayer(nn.Module):
    """ Single Transformer Decoder Layer (Non-Autoregressive Style) """
    def __init__(self, hidden_dim, n_heads, filter_size, kernel_size, dropout):
        super().__init__()
        # In non-autoregressive models like FastSpeech 2, the decoder often uses
        # unmasked self-attention similar to the encoder.
        self.self_attn = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, filter_size, kernel_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        """
        Args:
            tgt (Tensor): Input tensor (Batch, SeqLen, HiddenDim)
            tgt_mask (Tensor, optional): Mask for self-attention (Batch, 1, SeqLen) or (Batch, SeqLen, SeqLen). Defaults to None.
        Returns:
            Tensor: Output tensor (Batch, SeqLen, HiddenDim)
        """
        # Self-Attention + Add & Norm
        attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))

        # Feed-Forward + Add & Norm
        ffn_output = self.ffn(tgt)
        tgt = self.norm2(tgt + self.dropout(ffn_output))
        return tgt

class VariancePredictor(nn.Module):
    """ Predicts Duration, Pitch, or Energy """
    def __init__(self, hidden_dim, predictor_hidden_dim, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, predictor_hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(predictor_hidden_dim)
        self.conv2 = nn.Conv1d(predictor_hidden_dim, predictor_hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(predictor_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(predictor_hidden_dim, 1) # Predict a single value per frame/phoneme

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor (Batch, SeqLen, HiddenDim)
            mask (Tensor, optional): Boolean mask (Batch, SeqLen). Defaults to None.
        Returns:
            Tensor: Predicted values (Batch, SeqLen, 1)
        """
        # Transpose for Conv1D: (Batch, HiddenDim, SeqLen)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0) # Mask padding before conv

        x = F.relu(self.conv1(x))
        # Transpose for LayerNorm: (Batch, SeqLen, HiddenDim)
        x = self.norm1(x.transpose(1, 2))
        x = self.dropout(x)
        # Transpose back for Conv1D: (Batch, HiddenDim, SeqLen)
        x = F.relu(self.conv2(x.transpose(1, 2)))
        # Transpose for LayerNorm: (Batch, SeqLen, HiddenDim)
        x = self.norm2(x.transpose(1, 2))
        x = self.dropout(x)

        # Linear projection
        output = self.linear(x) # (Batch, SeqLen, 1)

        if mask is not None:
            output = output.masked_fill(~mask.unsqueeze(-1), 0.0) # Mask padding after linear

        return output

class LengthRegulator(nn.Module):
    """ Expands sequence based on durations """
    def repeat_interleave(self, x, repeats):
        """
        Torch repeat_interleave is slow on CUDA. This is a faster alternative.
        Args:
            x (Tensor): Input tensor (Batch, SeqLen, Dim)
            repeats (Tensor): Durations (Batch, SeqLen) - must be LongTensor
        Returns:
            Tensor: Expanded tensor (Batch, TotalLen, Dim), Lengths (Batch,)
        """
        batch_size, seq_len, dim = x.shape
        total_lengths = repeats.sum(dim=1)
        max_len = total_lengths.max().item()
        output = torch.zeros(batch_size, max_len, dim, device=x.device, dtype=x.dtype)
        # Create indices for scattering
        end_indices = torch.cumsum(repeats, dim=1) # (B, SL)
        start_indices = end_indices - repeats       # (B, SL)

        for i in range(batch_size):
            count = 0
            for j in range(seq_len):
                duration = repeats[i, j].item()
                if duration > 0:
                    # Slice the output tensor and assign the repeated value
                    output[i, count:count+duration, :] = x[i, j, :].unsqueeze(0)
                    count += duration
        return output, total_lengths

    def forward(self, x, durations, duration_scale=1.0):
        """
        Args:
            x (Tensor): Input tensor (Batch, SeqLen, HiddenDim)
            durations (Tensor): Duration tensor (Batch, SeqLen) - can be float from predictor
            duration_scale (float): Scale factor for durations (e.g., for speed control). Defaults to 1.0.
        Returns:
            Tensor: Expanded tensor (Batch, TotalLength, HiddenDim)
            Tensor: Mel lengths (Batch,)
        """
        # Round durations and ensure they are positive integers
        durations = (durations * duration_scale + 0.5).long()
        durations = torch.clamp(durations, min=1)

        expanded_x, mel_lengths = self.repeat_interleave(x, durations)
        return expanded_x, mel_lengths

# --- Main Modules ---

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        transformer_config = model_config['transformer']
        self.hidden_dim = transformer_config['hidden_dim']
        self.embedding_dim = model_config['embedding_dim'] # Phoneme embedding dim

        # Input projection if embedding_dim != hidden_dim
        if self.embedding_dim != self.hidden_dim:
             self.input_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        else:
             self.input_proj = nn.Identity()

        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=self.hidden_dim,
                n_heads=transformer_config['encoder_heads'],
                filter_size=transformer_config['conv_filter_size'],
                kernel_size=transformer_config['conv_kernel_size'],
                dropout=transformer_config['dropout']
            ) for _ in range(transformer_config['encoder_layers'])
        ])
        self.dropout = nn.Dropout(transformer_config['dropout'])

    def forward(self, x, src_mask):
        """
        Args:
            x (Tensor): Input phoneme embeddings (Batch, SeqLen, EmbDim)
            src_mask (Tensor): Boolean source mask (Batch, SeqLen)
        Returns:
            Tensor: Encoder output (Batch, SeqLen, HiddenDim)
        """
        x = self.input_proj(x) # Project to hidden_dim
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Create attention mask (Batch, 1, SeqLen) or (Batch, SeqLen, SeqLen)
        # For self-attention, we usually need (Batch, SeqLen, SeqLen) or allow broadcasting from (B, 1, SL)
        attn_mask = src_mask.unsqueeze(1) # (Batch, 1, SeqLen)

        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)

        # Apply mask again before returning? Optional, depends on predictor needs.
        if src_mask is not None:
             x = x.masked_fill(~src_mask.unsqueeze(-1), 0.0)

        return x

class VarianceAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        va_config = model_config['variance_adaptor']
        transformer_config = model_config['transformer']
        self.hidden_dim = transformer_config['hidden_dim']
        predictor_hidden = va_config['hidden_dim']
        kernel_size = va_config['kernel_size']
        dropout = va_config['dropout']

        self.duration_predictor = VariancePredictor(self.hidden_dim, predictor_hidden, kernel_size, dropout)
        self.pitch_predictor = VariancePredictor(self.hidden_dim, predictor_hidden, kernel_size, dropout)
        self.energy_predictor = VariancePredictor(self.hidden_dim, predictor_hidden, kernel_size, dropout)
        self.length_regulator = LengthRegulator()

        # Embeddings for pitch and energy (assuming continuous for now, adjust if quantized)
        # Project pitch/energy to hidden_dim before adding
        self.pitch_embedding = nn.Linear(1, self.hidden_dim)
        self.energy_embedding = nn.Linear(1, self.hidden_dim)

    def forward(self, x, src_mask, mel_mask=None,
                gt_duration=None, gt_pitch=None, gt_energy=None,
                duration_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        """
        Args:
            x (Tensor): Encoder output (Batch, SrcLen, HiddenDim)
            src_mask (Tensor): Source mask (Batch, SrcLen)
            mel_mask (Tensor, optional): Target mel mask (Batch, MelLen). Used during training for predictors. Defaults to None.
            gt_duration (Tensor, optional): Ground truth duration (Batch, SrcLen). Defaults to None.
            gt_pitch (Tensor, optional): Ground truth pitch (Batch, MelLen, 1). Defaults to None.
            gt_energy (Tensor, optional): Ground truth energy (Batch, MelLen, 1). Defaults to None.
            duration_scale (float): Duration scale for inference. Defaults to 1.0.
            pitch_scale (float): Pitch scale for inference. Defaults to 1.0.
            energy_scale (float): Energy scale for inference. Defaults to 1.0.

        Returns:
            Tensor: Variance adaptor output (Batch, MelLen, HiddenDim)
            Tensor: Predicted log duration (Batch, SrcLen, 1)
            Tensor: Predicted pitch (Batch, MelLen, 1)
            Tensor: Predicted energy (Batch, MelLen, 1)
            Tensor: Mel lengths after length regulation (Batch,)
        """
        # Predict duration (log scale is common)
        # Stop gradient to duration predictor as in FastSpeech 2
        log_duration_pred = self.duration_predictor(x.detach(), src_mask)

        # Use GT duration for training, predicted for inference
        if gt_duration is not None: # Training
            duration_used = gt_duration.float() # Ensure float for LR
            # Clamp GT durations to avoid issues if preprocessing yielded zeros/negatives
            duration_used = torch.clamp(duration_used, min=1.0)
        else: # Inference
            # Convert log duration prediction to linear scale duration
            duration_pred = torch.exp(log_duration_pred) - 1 # Assuming predictor predicts log(dur+1)
            duration_pred = torch.clamp(duration_pred, min=0) # Ensure non-negative
            duration_used = duration_pred.squeeze(-1) # Remove last dim: (B, SrcLen)

        # Length Regulator
        # Input x: (B, SrcLen, H), duration_used: (B, SrcLen)
        expanded_x, mel_lengths = self.length_regulator(x, duration_used, duration_scale)
        # expanded_x: (B, MelLen, H)

        # Create mel_mask based on actual expanded lengths for internal use
        current_mel_mask = self.create_mask(mel_lengths, expanded_x.size(1)) # (B, MelLen_actual)

        # Predict Pitch and Energy using expanded sequence and its corresponding mask
        pitch_pred = self.pitch_predictor(expanded_x, current_mel_mask)   # (B, MelLen_actual, 1)
        energy_pred = self.energy_predictor(expanded_x, current_mel_mask) # (B, MelLen_actual, 1)

        # Use GT pitch/energy for training, predicted for inference
        # Ensure GTs are reshaped to match expanded_x length before embedding
        max_len_actual = expanded_x.size(1)

        if gt_pitch is not None: # Training
            # gt_pitch is padded to max_mel_len by collate_fn
            pitch_to_embed = torch.zeros_like(expanded_x[..., 0:1]) # (B, MelLen_actual, 1)
            len_to_copy_pitch = min(gt_pitch.size(1), max_len_actual)
            pitch_to_embed[:, :len_to_copy_pitch, :] = gt_pitch[:, :len_to_copy_pitch, :]
        else: # Inference
            pitch_to_embed = pitch_pred * pitch_scale # Already (B, MelLen_actual, 1)

        if gt_energy is not None: # Training
            # gt_energy is padded to max_mel_len by collate_fn
            energy_to_embed = torch.zeros_like(expanded_x[..., 0:1]) # (B, MelLen_actual, 1)
            len_to_copy_energy = min(gt_energy.size(1), max_len_actual)
            energy_to_embed[:, :len_to_copy_energy, :] = gt_energy[:, :len_to_copy_energy, :]
        else: # Inference
            energy_to_embed = energy_pred * energy_scale # Already (B, MelLen_actual, 1)

        # Add pitch and energy embeddings - Now dimensions should match
        pitch_emb = self.pitch_embedding(pitch_to_embed)   # (B, MelLen_actual, H)
        energy_emb = self.energy_embedding(energy_to_embed) # (B, MelLen_actual, H)

        output = expanded_x + pitch_emb + energy_emb # (B, MelLen_actual, H)

        # Apply the mask corresponding to the actual output length
        output = output.masked_fill(~current_mel_mask.unsqueeze(-1), 0.0)

        # Return predictions along with output and the correct mask for the decoder
        return output, log_duration_pred, pitch_pred, energy_pred, mel_lengths, current_mel_mask

    def create_mask(self, lengths, max_len=None):
        """ Helper to create mask within the module """
        if max_len is None:
            max_len = lengths.max().item()
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
        mask = ids < lengths.unsqueeze(1)
        return mask

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        transformer_config = model_config['transformer']
        self.hidden_dim = transformer_config['hidden_dim']

        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim=self.hidden_dim,
                n_heads=transformer_config['decoder_heads'],
                filter_size=transformer_config['conv_filter_size'],
                kernel_size=transformer_config['conv_kernel_size'],
                dropout=transformer_config['dropout']
            ) for _ in range(transformer_config['decoder_layers'])
        ])
        self.dropout = nn.Dropout(transformer_config['dropout'])
        self.output_linear = nn.Linear(self.hidden_dim, model_config['mel_bins'])

    def forward(self, x, mel_mask):
        """
        Args:
            x (Tensor): Input from Variance Adaptor (Batch, MelLen, HiddenDim)
            mel_mask (Tensor): Boolean mel mask (Batch, MelLen)
        Returns:
            Tensor: Predicted Mel Spectrogram (Batch, MelLen, MelBins)
        """
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Create attention mask (Batch, 1, MelLen) or (Batch, MelLen, MelLen)
        attn_mask = mel_mask.unsqueeze(1) # (Batch, 1, MelLen)

        for layer in self.layers:
            x = layer(x, tgt_mask=attn_mask)

        # Apply mask before final projection
        if mel_mask is not None:
             x = x.masked_fill(~mel_mask.unsqueeze(-1), 0.0)

        # Project to Mel bins
        output = self.output_linear(x) # (Batch, MelLen, MelBins)

        # Apply mask again? Usually done before projection.
        if mel_mask is not None:
             output = output.masked_fill(~mel_mask.unsqueeze(-1), 0.0)

        return output

# --- Lightning Module ---

class TransformerSVS(pl.LightningModule):
    """ Transformer-based SVS model (FastSpeech 2 style) """
    def __init__(self, config, vocab_size): # Pass vocab_size explicitly
        super().__init__()
        self.save_hyperparameters(config) # Saves config to hparams
        self.config = config
        self.vocab_size = vocab_size # Store vocab_size
        model_config = config['model']
        train_config = config['train']

        # --- Layers ---
        self.phoneme_embedding = nn.Embedding(
            self.vocab_size,
            model_config['embedding_dim'],
            padding_idx=0 # Assuming 0 is the padding index
        )
        self.encoder = TransformerEncoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = TransformerDecoder(config)

        # --- Loss Functions ---
        # Use reduction='none' for manual masking and averaging
        self.mel_loss_fn = nn.L1Loss(reduction='none') # Or MSELoss
        self.duration_loss_fn = nn.MSELoss(reduction='none')
        self.pitch_loss_fn = nn.MSELoss(reduction='none')
        self.energy_loss_fn = nn.MSELoss(reduction='none')

        # --- Loss Weights ---
        self.loss_weights = train_config.get('loss_weights', {
            'mel': 1.0, 'duration': 1.0, 'pitch': 1.0, 'energy': 1.0
        })

        # --- Training Params ---
        self.learning_rate = train_config['learning_rate']
        self.log_spec_every_n = train_config.get('log_spectrogram_every_n_val_steps', 50)

    def forward(self, phonemes, src_mask, mel_mask=None,
                gt_duration=None, gt_pitch=None, gt_energy=None,
                duration_scale=1.0, pitch_scale=1.0, energy_scale=1.0):
        """
        Forward pass for training and inference.

        Args:
            phonemes (Tensor): Phoneme IDs (Batch, SrcLen).
            src_mask (Tensor): Source mask (Batch, SrcLen).
            mel_mask (Tensor, optional): Target mel mask (Batch, MelLen). Required for training variance predictors.
            gt_duration (Tensor, optional): GT Duration (Batch, SrcLen). For training VA.
            gt_pitch (Tensor, optional): GT Pitch (Batch, MelLen, 1). For training VA.
            gt_energy (Tensor, optional): GT Energy (Batch, MelLen, 1). For training VA.
            duration_scale (float): For inference speed control.
            pitch_scale (float): For inference pitch control.
            energy_scale (float): For inference energy control.

        Returns:
            Tuple: Contains:
                - mel_output (Tensor): Predicted Mel (Batch, MelLen, MelBins)
                - log_duration_pred (Tensor): Predicted log duration (Batch, SrcLen, 1)
                - pitch_pred (Tensor): Predicted pitch (Batch, MelLen, 1)
                - energy_pred (Tensor): Predicted energy (Batch, MelLen, 1)
                - mel_lengths (Tensor): Lengths after duration regulator (Batch,)
                - mel_mask (Tensor): Generated or passed mel mask (Batch, MelLen)
        """
        # 1. Embed Phonemes
        embedded_phonemes = self.phoneme_embedding(phonemes) # (B, SrcLen, EmbDim)

        # 2. Encoder
        encoder_output = self.encoder(embedded_phonemes, src_mask) # (B, SrcLen, HiddenDim)

        # 3. Variance Adaptor
        # The variance adaptor now returns the correct mask for its output
        va_output, log_duration_pred, pitch_pred, energy_pred, mel_lengths, decoder_mask = self.variance_adaptor(
            encoder_output, src_mask, mel_mask=mel_mask, # Pass original mel_mask for GT variance handling
            gt_duration=gt_duration, gt_pitch=gt_pitch, gt_energy=gt_energy,
            duration_scale=duration_scale, pitch_scale=pitch_scale, energy_scale=energy_scale
        )
        # va_output: (B, MelLen_actual, HiddenDim)
        # decoder_mask: (B, MelLen_actual)

        # Note: The original mel_mask (from batch) is still needed for loss calculation later.
        # The decoder_mask is specifically for the decoder input.

        # 4. Decoder
        # Pass the output of the VA and its corresponding mask to the decoder
        mel_output = self.decoder(va_output, decoder_mask) # (B, MelLen_actual, MelBins)

        return mel_output, log_duration_pred, pitch_pred, energy_pred, mel_lengths, mel_mask

    def _calculate_loss(self, batch, predictions):
        """ Calculates all loss components. """
        mel_pred, log_duration_pred, pitch_pred, energy_pred, _, pred_mel_mask = predictions

        # --- Ground Truths & Masks ---
        gt_mel = batch['mel']           # (B, MelBins, MelLen) - from dataloader
        gt_duration = batch['duration'] # (B, SrcLen) - from dataloader
        gt_pitch = batch['f0']          # (B, MelLen, 1) - from dataloader
        gt_energy = batch['energy']     # (B, MelLen, 1) - from dataloader
        src_mask = batch['src_mask']    # (B, SrcLen) - from dataloader
        gt_mel_mask = batch['mel_mask'] # (B, MelLen) - from dataloader

        # Ensure GT mel is (B, MelLen, MelBins) for loss calculation
        if gt_mel.size(1) == self.config['model']['mel_bins']:
             gt_mel = gt_mel.transpose(1, 2)

        # --- Mel Loss ---
        # Ensure prediction matches GT length (should be handled by VA/masking)
        max_len_gt = gt_mel.size(1)
        mel_pred = mel_pred[:, :max_len_gt, :] # Truncate if needed

        # Apply mask (B, MelLen, MelBins) * (B, MelLen, 1)
        mel_loss_elementwise = self.mel_loss_fn(mel_pred, gt_mel)
        mel_loss_masked = mel_loss_elementwise.masked_fill(~gt_mel_mask.unsqueeze(-1), 0.0)
        mel_loss = mel_loss_masked.sum() / gt_mel_mask.sum() # Average over valid frames/bins

        # --- Duration Loss ---
        # Predictor outputs log(dur+1) -> target should be log(gt_dur+1)
        # Ensure gt_duration is float and >= 1
        gt_duration_log = torch.log(gt_duration.float().clamp(min=1.0))
        # Apply mask (B, SrcLen, 1) * (B, SrcLen, 1)
        duration_loss_elementwise = self.duration_loss_fn(log_duration_pred.squeeze(-1), gt_duration_log)
        duration_loss_masked = duration_loss_elementwise.masked_fill(~src_mask, 0.0)
        duration_loss = duration_loss_masked.sum() / src_mask.sum() # Average over valid phonemes

        # --- Pitch Loss ---
        # pitch_pred shape: (B, MelLen_actual, 1)
        # gt_pitch shape: (B, MelLen_padded, 1)
        # gt_mel_mask shape: (B, MelLen_padded)
        len_pred_pitch = pitch_pred.size(1)
        len_gt_pitch = gt_pitch.size(1)
        len_common_pitch = min(len_pred_pitch, len_gt_pitch)
        # Truncate tensors to common length
        pitch_pred_trunc = pitch_pred[:, :len_common_pitch, :]
        gt_pitch_trunc = gt_pitch[:, :len_common_pitch, :]
        # Use GT mask, truncated to common length
        mask_trunc_pitch = gt_mel_mask[:, :len_common_pitch].unsqueeze(-1) # (B, len_common, 1)
        # Calculate loss over common length using mask
        pitch_loss_elementwise = self.pitch_loss_fn(pitch_pred_trunc, gt_pitch_trunc)
        pitch_loss_masked = pitch_loss_elementwise.masked_fill(~mask_trunc_pitch, 0.0)
        pitch_loss = pitch_loss_masked.sum() / mask_trunc_pitch.sum() # Average over valid frames in common length

        # --- Energy Loss ---
        # energy_pred shape: (B, MelLen_actual, 1)
        # gt_energy shape: (B, MelLen_padded, 1)
        len_pred_energy = energy_pred.size(1)
        len_gt_energy = gt_energy.size(1)
        len_common_energy = min(len_pred_energy, len_gt_energy)
        # Truncate tensors to common length
        energy_pred_trunc = energy_pred[:, :len_common_energy, :]
        gt_energy_trunc = gt_energy[:, :len_common_energy, :]
        # Use GT mask, truncated to common length
        mask_trunc_energy = gt_mel_mask[:, :len_common_energy].unsqueeze(-1) # (B, len_common, 1)
        # Calculate loss over common length using mask
        energy_loss_elementwise = self.energy_loss_fn(energy_pred_trunc, gt_energy_trunc)
        energy_loss_masked = energy_loss_elementwise.masked_fill(~mask_trunc_energy, 0.0)
        energy_loss = energy_loss_masked.sum() / mask_trunc_energy.sum() # Average over valid frames in common length

        # --- Total Loss ---
        total_loss = (self.loss_weights['mel'] * mel_loss +
                      self.loss_weights['duration'] * duration_loss +
                      self.loss_weights['pitch'] * pitch_loss +
                      self.loss_weights['energy'] * energy_loss)

        losses = {
            'mel': mel_loss,
            'duration': duration_loss,
            'pitch': pitch_loss,
            'energy': energy_loss,
            'total': total_loss
        }
        return losses

    def training_step(self, batch, batch_idx):
        # Unpack batch elements needed for forward and loss
        phonemes = batch['phoneme']
        src_mask = batch['src_mask']
        mel_mask = batch['mel_mask']
        gt_duration = batch['duration']
        gt_pitch = batch['f0']
        gt_energy = batch['energy']

        # Forward pass using ground truths for variance adaptor inputs
        predictions = self(phonemes, src_mask, mel_mask=mel_mask,
                           gt_duration=gt_duration, gt_pitch=gt_pitch, gt_energy=gt_energy)

        # Calculate losses
        losses = self._calculate_loss(batch, predictions)

        # Logging
        self.log('train_loss', losses['total'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0))
        self.log('train_mel_loss', losses['mel'], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_duration_loss', losses['duration'], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_pitch_loss', losses['pitch'], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_energy_loss', losses['energy'], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))

        return losses['total']

    def validation_step(self, batch, batch_idx):
        # Unpack batch elements needed for forward and loss
        phonemes = batch['phoneme']
        src_mask = batch['src_mask']
        mel_mask = batch['mel_mask'] # Use GT mel mask for loss calculation
        gt_duration = batch['duration']
        gt_pitch = batch['f0']
        gt_energy = batch['energy']
        gt_mel = batch['mel'] # Needed for plotting


        # Forward pass - Use predicted variances for inference simulation, but GT mask for loss
        # Pass GTs to calculate loss correctly, but scales are 1.0
        predictions = self(phonemes, src_mask, mel_mask=mel_mask,
                           gt_duration=gt_duration, gt_pitch=gt_pitch, gt_energy=gt_energy) # Use GTs for loss calc

        # Calculate losses using GT masks
        losses = self._calculate_loss(batch, predictions)

        # Logging
        self.log('val_loss', losses['total'], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0))
        self.log('val_mel_loss', losses['mel'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('val_duration_loss', losses['duration'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('val_pitch_loss', losses['pitch'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('val_energy_loss', losses['energy'], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))

        # --- Log Spectrogram Comparison ---
        # Use the predicted mel from the forward pass (which used predicted variances internally if GTs weren't passed,
        # but here we passed GTs for loss calculation consistency. For true inference vis, call forward without GTs)
        mel_pred_for_plot = predictions[0] # (B, MelLen, MelBins)
        mel_lengths_for_plot = batch['mel_lengths'] # Use original lengths from collate_fn

        if self.logger and self.log_spec_every_n > 0 and batch_idx % self.log_spec_every_n == 0:
            try:
                # Select the first sample
                idx = 0
                first_sample_len = mel_lengths_for_plot[idx].item()
                gt_mel_sample = gt_mel[idx, :, :first_sample_len].transpose(0, 1).cpu().detach() # Transpose to (Freq, Time)
                # Prediction needs transpose: (ActualTime, MelBins) -> (MelBins, ActualTime)
                pred_mel_sample = mel_pred_for_plot[idx, :first_sample_len, :].transpose(0, 1).cpu().detach() # (MelBins, ActualTime)

                if gt_mel_sample.shape == pred_mel_sample.shape and gt_mel_sample.shape[1] > 0:
                    fig = plot_spectrograms_to_figure(
                        gt_mel_sample,
                        pred_mel_sample,
                        title=f"Epoch {self.current_epoch} Batch {batch_idx} - Spec Comp",
                        vmin=None, vmax=None # Auto-scale for now
                    )
                    self.logger.experiment.add_figure(
                        f"Validation_Images/Spectrogram_Comparison_Batch_{batch_idx}",
                        fig, global_step=self.global_step
                    )
                    plt.close(fig)
                else:
                     print(f"Warning: Skipping spectrogram plot for E{self.current_epoch} B{batch_idx} due to shape mismatch or zero length. GT: {gt_mel_sample.shape}, Pred: {pred_mel_sample.shape}")

            except Exception as e:
                print(f"Warning: Error generating/logging spectrogram plot for E{self.current_epoch} B{batch_idx}: {e}")

        return losses['total']

    def configure_optimizers(self):
        # AdamW is common for Transformers
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Use Noam scheduler
        warmup_steps = self.config['train'].get('warmup_steps', 4000) # Get from config

        def lr_lambda(step):
            # Adding 1 to step because Pytorch Lightning scheduler steps start from 0
            # Handle step 0 case for division
            step = step + 1
            if warmup_steps == 0: # Avoid division by zero if warmup is disabled
                 return 1.0 # Constant LR factor

            # Calculate learning rate factor based on step
            if step < warmup_steps:
                # Linear warmup phase
                return float(step) / float(warmup_steps)
            else:
                # Decay phase (inverse square root)
                # Factor should be relative to the LR at the end of warmup
                return (float(warmup_steps) / float(step)) ** 0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step', # Call scheduler every step
                'frequency': 1,
                'name': 'learning_rate' # Name for logging
            }
        }