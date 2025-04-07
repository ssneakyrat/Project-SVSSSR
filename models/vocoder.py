# models/vocoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

from utils.plotting import plot_spectrograms_to_figure

logger = logging.getLogger(__name__)

class ConditionNetwork(nn.Module):
    """
    Condition Network: Processes mel spectrogram input to create conditioning vectors.
    """
    def __init__(self, mel_bins=80, hidden_dims=64, out_dims=64):
        super().__init__()
        self.conv1 = nn.Conv1d(mel_bins, hidden_dims, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=3, padding=1)
        
        # Bidirectional GRU for temporal context
        self.bigru = nn.GRU(
            hidden_dims, 
            hidden_dims // 2,
            batch_first=True,
            bidirectional=True
        )
        
        # Two linear layers with tanh activation
        self.linear1 = nn.Linear(hidden_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, out_dims)
        
    def forward(self, mel):
        """
        Args:
            mel: [batch_size, time_steps, mel_bins]
        Returns:
            cond: [batch_size, time_steps, out_dims]
        """
        # Transpose for 1D convolution: [B, T, M] -> [B, M, T]
        x = mel.transpose(1, 2)
        
        # 1D CNN for local feature extraction
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        
        # Transpose back for GRU: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # Bidirectional GRU for temporal context
        x, _ = self.bigru(x)
        
        # Additional processing with linear layers
        x = torch.tanh(self.linear1(x))
        cond = torch.tanh(self.linear2(x))
        
        return cond

class SparseGRU(nn.Module):
    """
    Sparse GRU implementation using block-sparse patterns.
    Simulates 80% sparsity while maintaining efficiency.
    """
    def __init__(self, input_size, hidden_size, sparsity=0.8, block_size=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.block_size = block_size
        
        # Create a standard GRU cell
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        
        # Create sparse masks for the weights
        self._create_sparse_masks()
        
    def _create_sparse_masks(self):
        """Creates binary masks for weight sparsity"""
        # Calculate how many blocks we need
        ih_blocks_in = (self.input_size + self.block_size - 1) // self.block_size
        ih_blocks_out = (self.hidden_size + self.block_size - 1) // self.block_size
        hh_blocks_in = (self.hidden_size + self.block_size - 1) // self.block_size
        hh_blocks_out = (self.hidden_size + self.block_size - 1) // self.block_size
        
        # Total number of blocks
        ih_total_blocks = ih_blocks_in * ih_blocks_out * 3  # 3 for reset, update, new gates
        hh_total_blocks = hh_blocks_in * hh_blocks_out * 3  # 3 for reset, update, new gates
        
        # Number of blocks to keep (1 - sparsity)
        ih_blocks_to_keep = int(ih_total_blocks * (1 - self.sparsity))
        hh_blocks_to_keep = int(hh_total_blocks * (1 - self.sparsity))
        
        # Sample blocks to keep
        # Fixed random seed for reproducibility
        rng = np.random.RandomState(42)
        self.ih_keep_indices = rng.choice(ih_total_blocks, ih_blocks_to_keep, replace=False)
        self.hh_keep_indices = rng.choice(hh_total_blocks, hh_blocks_to_keep, replace=False)
        
        # Register buffer for masks - PyTorch will handle device placement
        ih_mask = torch.zeros(self.gru_cell.weight_ih.shape)
        hh_mask = torch.zeros(self.gru_cell.weight_hh.shape)
        
        # Fill the masks with 1s for blocks to keep
        for idx in self.ih_keep_indices:
            gate = idx // (ih_blocks_in * ih_blocks_out)
            remain = idx % (ih_blocks_in * ih_blocks_out)
            block_i = remain // ih_blocks_out
            block_j = remain % ih_blocks_out
            
            i_start = gate * self.hidden_size + block_j * self.block_size
            i_end = min(i_start + self.block_size, (gate + 1) * self.hidden_size)
            j_start = block_i * self.block_size
            j_end = min(j_start + self.block_size, self.input_size)
            
            if i_end > i_start and j_end > j_start:
                ih_mask[i_start:i_end, j_start:j_end] = 1
        
        for idx in self.hh_keep_indices:
            gate = idx // (hh_blocks_in * hh_blocks_out)
            remain = idx % (hh_blocks_in * hh_blocks_out)
            block_i = remain // hh_blocks_out
            block_j = remain % hh_blocks_out
            
            i_start = gate * self.hidden_size + block_j * self.block_size
            i_end = min(i_start + self.block_size, (gate + 1) * self.hidden_size)
            j_start = block_i * self.block_size
            j_end = min(j_start + self.block_size, self.hidden_size)
            
            if i_end > i_start and j_end > j_start:
                hh_mask[i_start:i_end, j_start:j_end] = 1
        
        self.register_buffer('ih_mask', ih_mask)
        self.register_buffer('hh_mask', hh_mask)
        
    def _apply_masks(self):
        """Apply masks to the weights"""
        self.gru_cell.weight_ih.data.mul_(self.ih_mask)
        self.gru_cell.weight_hh.data.mul_(self.hh_mask)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch_size, time_steps, input_size]
            hidden: Initial hidden state, optional
            
        Returns:
            outputs: [batch_size, time_steps, hidden_size]
        """
        batch_size, seq_len, _ = x.size()
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        
        # Before running, apply the sparsity masks
        self._apply_masks()
        
        # Process input sequence one step at a time
        for t in range(seq_len):
            hidden = self.gru_cell(x[:, t], hidden)
            outputs.append(hidden)
        
        # Stack outputs along time dimension
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, hidden

class DualSoftmaxOutput(nn.Module):
    """
    Dual softmax output layer for 16-bit audio (8 bits coarse + 8 bits fine).
    """
    def __init__(self, hidden_size, bits=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.bits = bits
        self.num_classes = 2**bits  # 256 for 8 bits
        
        # Output projection layers
        self.coarse_proj = nn.Linear(hidden_size, self.num_classes)
        self.fine_proj = nn.Linear(hidden_size + self.num_classes, self.num_classes)
        
    def forward(self, x, coarse_x=None):
        """
        Args:
            x: [batch_size, time_steps, hidden_size]
            coarse_x: Coarse categorical input for conditioning the fine output (during inference),
                      shape: [batch_size, time_steps], optional
        
        Returns:
            coarse_logits: [batch_size, time_steps, num_classes]
            fine_logits: [batch_size, time_steps, num_classes]
        """
        batch_size, seq_len, _ = x.size()
        
        # Coarse prediction
        coarse_logits = self.coarse_proj(x)
        
        if self.training or coarse_x is None:
            # During training, use predicted coarse logits
            coarse_probs = F.softmax(coarse_logits, dim=-1)
        else:
            # During inference, use provided coarse values
            coarse_probs = F.one_hot(coarse_x, num_classes=self.num_classes).float()
        
        # Concatenate hidden state with coarse probabilities
        fine_input = torch.cat([x, coarse_probs], dim=-1)
        
        # Fine prediction
        fine_logits = self.fine_proj(fine_input)
        
        return coarse_logits, fine_logits
    
    def sample(self, x, temperature=1.0):
        """
        Sample audio values from logits with temperature control.
        
        Args:
            x: [batch_size, time_steps, hidden_size]
            temperature: Value to control randomness (lower = more deterministic)
            
        Returns:
            coarse_samples: [batch_size, time_steps]
            fine_samples: [batch_size, time_steps]
            audio_samples: [batch_size, time_steps] - Combined 16-bit audio
        """
        batch_size, seq_len, _ = x.size()
        
        coarse_logits = self.coarse_proj(x)
        
        if temperature > 0:
            # Apply temperature scaling
            coarse_logits = coarse_logits / temperature
            
            # Sample from scaled logits
            coarse_probs = F.softmax(coarse_logits, dim=-1)
            coarse_dist = torch.distributions.Categorical(probs=coarse_probs)
            coarse_samples = coarse_dist.sample()
        else:
            # Deterministic (argmax)
            coarse_samples = torch.argmax(coarse_logits, dim=-1)
        
        # One-hot encode coarse samples for fine prediction
        coarse_one_hot = F.one_hot(coarse_samples, num_classes=self.num_classes).float()
        
        # Concatenate for fine prediction
        fine_input = torch.cat([x, coarse_one_hot], dim=-1)
        fine_logits = self.fine_proj(fine_input)
        
        if temperature > 0:
            # Apply temperature scaling
            fine_logits = fine_logits / temperature
            
            # Sample from scaled logits
            fine_probs = F.softmax(fine_logits, dim=-1)
            fine_dist = torch.distributions.Categorical(probs=fine_probs)
            fine_samples = fine_dist.sample()
        else:
            # Deterministic (argmax)
            fine_samples = torch.argmax(fine_logits, dim=-1)
        
        # Combine coarse and fine samples into 16-bit audio
        audio_samples = (coarse_samples.long() << self.bits) + fine_samples.long()
        
        return coarse_samples, fine_samples, audio_samples

class TinyWaveRNN(nn.Module):
    """
    Tiny WaveRNN model with sparse GRU and dual softmax output.
    """
    def __init__(
    self,
    mel_bins=80,
    conditioning_dims=64,
    gru_dims=256,
    gru_sparsity=0.8,
    gru_block_size=16,
    bits=8,
    use_f0=True,
    f0_dims=16,
    use_unvoiced=True,
    unvoiced_dims=16
):
        super().__init__()
        
        self.bits = bits
        self.use_f0 = use_f0
        self.use_unvoiced = use_unvoiced
        
        # Calculate total input dimensions for mel processing
        total_input_dims = mel_bins
        
        # Calculate the FINAL conditioning dimensions including all features
        final_conditioning_dims = conditioning_dims
        if use_f0:
            total_input_dims += f0_dims
            self.f0_embedding = nn.Linear(1, f0_dims)
            final_conditioning_dims += f0_dims
            
        if use_unvoiced:
            total_input_dims += unvoiced_dims
            self.unvoiced_embedding = nn.Linear(1, unvoiced_dims)
            final_conditioning_dims += unvoiced_dims
        
        # Store the final conditioning dimensions
        self.conditioning_dims = final_conditioning_dims
                
        # Condition network now correctly outputs the final conditioning dimensions
        self.condition_network = ConditionNetwork(
            mel_bins=total_input_dims,
            hidden_dims=conditioning_dims,
            out_dims=final_conditioning_dims  # This is the key change!
        )
        
        # GRU input size: previous sample + conditioning
        gru_input_size = 1 + final_conditioning_dims
        
        # Sparse GRU layer
        self.sparse_gru = SparseGRU(
            input_size=gru_input_size,
            hidden_size=gru_dims,
            sparsity=gru_sparsity,
            block_size=gru_block_size
        )
        
        # Dual softmax output for coarse and fine bits
        self.output_layer = DualSoftmaxOutput(
            hidden_size=gru_dims,
            bits=bits
        )
        
        # For u-law encoding/decoding
        self.register_buffer('ulaw_table', self._create_ulaw_table())
        self.register_buffer('inv_ulaw_table', self._create_inv_ulaw_table())
        
    def _create_ulaw_table(self):
        """Create the μ-law encoding table for audio quantization."""
        x = np.linspace(-1.0, 1.0, self.bits)
        mu = self.bits - 1
        
        # Convert to tensor before applying log
        x_tensor = torch.tensor(x)
        mu_tensor = torch.tensor(float(mu))
        
        # Fix: Ensure all values are tensors for PyTorch operations
        y = torch.sign(x_tensor) * torch.log(1 + mu_tensor * torch.abs(x_tensor)) / torch.log(torch.tensor(1 + mu))
        
        # Scale to [0, bits-1]
        y = ((y + 1) / 2 * (self.bits - 1)).round()
        return y
    
    def _create_inv_ulaw_table(self):
        """Create lookup table for u-law decoding."""
        table = torch.zeros(256)
        mu = 255.0
        
        for i in range(256):
            # Scale to [-1, 1]
            y = 2 * (i / 255) - 1
            # Apply inverse μ-law formula
            x = torch.sign(torch.tensor(y)) * (1 / mu) * ((1 + mu)**torch.abs(torch.tensor(y)) - 1)
            # Scale to [0, 2^16-1]
            pcm_val = ((x + 1) / 2 * (2**16 - 1)).round()
            table[i] = pcm_val
            
        return table
    
    def encode_ulaw(self, x):
        """Encode 16-bit PCM to 8-bit μ-law."""
        # Ensure input is within range [0, 2^16-1]
        x = x.clamp(0, 2**16 - 1)
        # Use lookup table for encoding
        return self.ulaw_table[x.long()].long()
    
    def decode_ulaw(self, y):
        """Decode 8-bit μ-law to 16-bit PCM."""
        # Ensure input is within range [0, 255]
        y = y.clamp(0, 255)
        # Use lookup table for decoding
        return self.inv_ulaw_table[y.long()].long()
        
    def forward(self, batch, return_hidden=False):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing:
                - mel_spec: [batch_size, time_steps, mel_bins]
                - target_audio: [batch_size, time_steps*hop_length]
                - f0: [batch_size, time_steps, 1] (optional)
                - unvoiced_flag: [batch_size, time_steps, 1] (optional)
            return_hidden: Whether to return hidden state
            
        Returns:
            outputs: Dictionary containing model outputs
        """
        mel = batch['mel_spec']
        batch_size, mel_len, mel_bins = mel.shape
        
        # Create conditioning input
        cond_input = mel
        
        if self.use_f0 and 'f0' in batch:
            f0 = batch['f0']
            f0_embed = self.f0_embedding(f0)
            cond_input = torch.cat([cond_input, f0_embed], dim=-1)
            
        if self.use_unvoiced and 'unvoiced_flag' in batch:
            unvoiced = batch['unvoiced_flag'].unsqueeze(-1)
            unvoiced_embed = self.unvoiced_embedding(unvoiced)
            cond_input = torch.cat([cond_input, unvoiced_embed], dim=-1)
            
        # Process through condition network
        conditioning = self.condition_network(cond_input)
        
        # Prepare audio target for training
        if 'target_audio' in batch:
            audio = batch['target_audio']
            
            # Reduce audio resolution to match hop_length
            hop_length = audio.shape[1] // mel_len
            
            # Take center samples from each hop
            audio_frames = audio[:, hop_length//2::hop_length]
            
            # Truncate to match mel length (in case of rounding issues)
            audio_frames = audio_frames[:, :mel_len]
            
            # Apply μ-law encoding - using a different approach
            # Remember device and shape for later
            device = audio_frames.device
            orig_shape = audio_frames.shape
            
            # Convert to numpy for safer bit operations
            audio_np = audio_frames.detach().cpu().numpy()
            
            # Convert from signed to unsigned (add 2^15)
            audio_np = audio_np.astype(np.int32) + (1 << 15)
            
            # Clamp to valid range 
            audio_np = np.clip(audio_np, 0, (1 << 16) - 1)
            
            # Apply μ-law encoding (this would use the ulaw_table defined elsewhere)
            # For simplicity, let's implement it directly here
            mu = 255.0
            # Scale to [-1, 1]
            x = (audio_np / (1 << 15)) - 1.0
            # Apply μ-law formula
            y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
            # Scale to [0, 255]
            audio_ulaw_np = np.round((y + 1) / 2 * 255).astype(np.uint8)
            
            # Split into coarse (high 4 bits) and fine (low 4 bits) components
            audio_coarse_np = audio_ulaw_np >> 4  # High 4 bits (0-15)
            audio_fine_np = audio_ulaw_np & 0x0F  # Low 4 bits (0-15)
            
            # Convert back to PyTorch tensors and move to original device
            audio_coarse = torch.from_numpy(audio_coarse_np).long().to(device)
            audio_fine = torch.from_numpy(audio_fine_np).long().to(device)
            
            # For training, we use the previous sample as input
            # Prepare previous frames for input (shift right, add zeros at start)
            prev_audio = torch.zeros_like(audio_frames)
            prev_audio[:, 1:] = audio_frames[:, :-1]
            
            # Prepare GRU inputs: previous sample + conditioning
            prev_audio_scaled = prev_audio.unsqueeze(-1) / (1 << 15)  # Scale to [-1, 1]
            gru_input = torch.cat([prev_audio_scaled, conditioning], dim=-1)
            
            # Process through sparse GRU
            gru_output, hidden = self.sparse_gru(gru_input)
            
            # Get coarse and fine predictions
            coarse_logits, fine_logits = self.output_layer(gru_output)
            
            outputs = {
                'coarse_logits': coarse_logits,
                'fine_logits': fine_logits,
                'audio_coarse': audio_coarse,
                'audio_fine': audio_fine,
            }
            
            if return_hidden:
                outputs['hidden'] = hidden
                
            return outputs
            
        else:
            # Inference mode - generate audio sample by sample
            # Start with silence
            audio_sample = torch.zeros(batch_size, 1, device=mel.device)
            generated_samples = []
            hidden = None
            
            # Generate one sample at a time for each frame
            for t in range(mel_len):
                frame_cond = conditioning[:, t:t+1, :]  # Get conditioning for this frame
                
                # Concatenate current audio sample with conditioning
                audio_scaled = audio_sample.unsqueeze(-1) / (1 << 15)  # Scale to [-1, 1]
                gru_input = torch.cat([audio_scaled, frame_cond], dim=-1)
                
                # Run through GRU
                gru_output, hidden = self.sparse_gru(gru_input, hidden)
                
                # Sample from output distribution
                coarse_sample, fine_sample, _ = self.output_layer.sample(gru_output, temperature=0.6)
                
                # Convert to audio sample (16-bit) using numpy
                coarse_np = coarse_sample.detach().cpu().numpy()
                fine_np = fine_sample.detach().cpu().numpy()
                
                # Combine coarse and fine components
                combined_np = (coarse_np << 4) | fine_np
                
                # Apply inverse μ-law
                # Scale to [-1, 1]
                y = (combined_np.astype(np.float32) / 255.0) * 2 - 1
                # Apply inverse μ-law formula
                mu = 255.0
                x = np.sign(y) * (1/mu) * ((1 + mu)**np.abs(y) - 1)
                # Scale to [0, 2^16-1]
                audio_16bit_np = ((x + 1) / 2 * ((1 << 16) - 1)).astype(np.int32)
                # Convert back to signed
                audio_sample_np = audio_16bit_np - (1 << 15)
                
                # Convert back to torch tensor
                audio_sample = torch.from_numpy(audio_sample_np).float().to(mel.device)
                
                generated_samples.append(audio_sample)
                
            # Concatenate all generated samples
            generated_audio = torch.stack(generated_samples, dim=1).squeeze(-1)
            
            return {'generated_audio': generated_audio}

class VocoderModel(pl.LightningModule):
    """
    PyTorch Lightning module for the Tiny WaveRNN vocoder.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Extract relevant configurations
        vocoder_config = config['vocoder']
        audio_config = config['audio']
        train_config = config['train']
        
        # Initialize model
        self.model = TinyWaveRNN(
            mel_bins=audio_config['n_mels'],
            conditioning_dims=64,
            gru_dims=256,
            gru_sparsity=0.8,
            gru_block_size=16,
            bits=8,
            use_f0=True,
            f0_dims=16,
            use_unvoiced=True,
            unvoiced_dims=16
        )
        
        # Loss weights
        self.loss_lambda_sc = train_config.get('loss_lambda_sc', 0.5)
        self.loss_lambda_mag = train_config.get('loss_lambda_mag', 1.0)
        self.loss_lambda_td = train_config.get('loss_lambda_td', 1.0)
        
        # STFT loss parameters
        self.fft_sizes = train_config.get('stft_loss_fft_sizes', [1024, 2048, 512])
        self.hop_sizes = train_config.get('stft_loss_hop_sizes', [120, 240, 50])
        self.win_lengths = train_config.get('stft_loss_win_lengths', [600, 1200, 240])
        
        # Sampling rate
        self.sample_rate = audio_config['sample_rate']
        
        # Logging
        self.log_interval = train_config.get('log_interval', 50)
        self.log_audio_epoch_interval = train_config.get('log_vocoder_audio_epoch_interval', 1)
        self.log_n_samples = train_config.get('log_n_vocoder_audio_samples', 1)
        self.log_spectrograms = train_config.get('log_vocoder_spectrograms', True)
        
    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)
    
    def _compute_loss(self, outputs, batch):
        """Compute training loss."""
        # Cross-entropy loss for coarse and fine components
        coarse_logits = outputs['coarse_logits']
        fine_logits = outputs['fine_logits']
        
        audio_coarse = outputs['audio_coarse']
        audio_fine = outputs['audio_fine']
        
        # Reshape tensors for loss calculation
        coarse_logits = coarse_logits.reshape(-1, coarse_logits.size(-1))
        fine_logits = fine_logits.reshape(-1, fine_logits.size(-1))
        audio_coarse = audio_coarse.reshape(-1)
        audio_fine = audio_fine.reshape(-1)
        
        # Calculate cross-entropy losses
        coarse_loss = F.cross_entropy(coarse_logits, audio_coarse)
        fine_loss = F.cross_entropy(fine_logits, audio_fine)
        
        # Combine losses (typically equal weight)
        total_loss = 0.5 * coarse_loss + 0.5 * fine_loss
        
        # Phase coherency loss for voiced regions (optional)
        if 'voiced_mask' in batch:
            voiced_mask = batch['voiced_mask']
            if voiced_mask.sum() > 0:
                # Add loss term focused on periodicity in voiced regions
                # (Implementation details would depend on specific approach)
                # For now, we'll just use the basic loss
                pass
        
        return {
            'loss': total_loss,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss
        }
        
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(batch)
        losses = self._compute_loss(outputs, batch)
        
        # Log losses
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/coarse_loss', losses['coarse_loss'], on_step=True, on_epoch=True)
        self.log('train/fine_loss', losses['fine_loss'], on_step=True, on_epoch=True)
        
        # Log samples at intervals
        if batch_idx % self.log_interval == 0:
            self._log_audio_samples(batch, outputs, prefix='train', max_samples=1)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(batch)
        losses = self._compute_loss(outputs, batch)
        
        # Log losses
        self.log('val/loss', losses['loss'], on_epoch=True, prog_bar=True)
        self.log('val/coarse_loss', losses['coarse_loss'], on_epoch=True)
        self.log('val/fine_loss', losses['fine_loss'], on_epoch=True)
        
        # Log samples at intervals
        if batch_idx == 0 and self.current_epoch % self.log_audio_epoch_interval == 0:
            self._log_audio_samples(batch, outputs, prefix='val', max_samples=self.log_n_samples)
            
        return losses['loss']
    
    def _log_audio_samples(self, batch, outputs, prefix='val', max_samples=1):
        """Log audio samples and spectrograms to TensorBoard."""
        if not self.logger:
            return
            
        # Get ground truth audio
        if 'target_audio' in batch:
            audio_gt = batch['target_audio']
            batch_size = audio_gt.shape[0]
            num_samples = min(batch_size, max_samples)
            
            # Process at most max_samples samples
            for i in range(num_samples):
                # Log ground truth audio
                self.logger.experiment.add_audio(
                    f'{prefix}/audio_gt_{i}',
                    audio_gt[i].detach().cpu().float().numpy() / 32768.0,
                    self.global_step,
                    sample_rate=self.sample_rate
                )
                
                # Generate audio for comparison
                inference_batch = {
                    'mel_spec': batch['mel_spec'][i:i+1],
                    'f0': batch['f0'][i:i+1] if 'f0' in batch else None,
                    'unvoiced_flag': batch['unvoiced_flag'][i:i+1] if 'unvoiced_flag' in batch else None
                }
                
                # Generate audio
                with torch.no_grad():
                    gen_output = self.model(inference_batch)
                
                if 'generated_audio' in gen_output:
                    gen_audio = gen_output['generated_audio'][0]
                    
                    # Log generated audio
                    self.logger.experiment.add_audio(
                        f'{prefix}/audio_gen_{i}',
                        gen_audio.detach().cpu().float().numpy() / 32768.0,
                        self.global_step,
                        sample_rate=self.sample_rate
                    )
                    
                    # Log spectrograms if enabled
                    if self.log_spectrograms:
                        # Create mel spectrograms for GT and generated audio
                        mel_gt = self._audio_to_mel(audio_gt[i].detach().cpu().numpy())
                        mel_gen = self._audio_to_mel(gen_audio.detach().cpu().numpy())
                        
                        # Create figure
                        fig = plot_spectrograms_to_figure(
                            ground_truth=torch.from_numpy(mel_gt),
                            prediction=torch.from_numpy(mel_gen),
                            title=f"Mel Spectrogram Comparison"
                        )
                        
                        # Log figure
                        self.logger.experiment.add_figure(
                            f'{prefix}/spectrogram_{i}',
                            fig,
                            self.global_step
                        )
    
    def _audio_to_mel(self, audio):
        """Convert audio to mel spectrogram for visualization."""
        try:
            import librosa
            # Use librosa to compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.config['audio']['n_fft'],
                hop_length=self.config['audio']['hop_length'],
                win_length=self.config['audio']['win_length'],
                n_mels=self.config['audio']['n_mels'],
                fmin=self.config['audio']['fmin'],
                fmax=self.config['audio']['fmax']
            )
            # Convert to log scale
            mel = librosa.power_to_db(mel, ref=np.max)
            return mel
        except ImportError:
            logger.warning("Librosa not available for mel spectrogram conversion.")
            # Return dummy spectrogram if librosa not available
            return np.zeros((self.config['audio']['n_mels'], 128))
                
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        lr = self.config['train'].get('vocoder_learning_rate', 0.0001)
        weight_decay = self.config['train'].get('weight_decay', 0.0001)
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        lr_factor = self.config['train'].get('lr_factor', 0.5)
        lr_patience = self.config['train'].get('lr_patience', 10)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch'
            }
        }