# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for mask creation
import pytorch_lightning as pl
import numpy as np # Needed for repeat_interleave calculation
import matplotlib.pyplot as plt
from utils.plotting import plot_spectrograms_to_figure

# +++++++++++++++++ PostNet Definition +++++++++++++++++
class PostNet(nn.Module):
    """Post-Net for refining predicted Mel spectrograms."""
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        postnet_config = model_config['postnet']
        mel_bins = model_config['mel_bins']
        n_convs = postnet_config['n_convs']
        channels = postnet_config['channels']
        kernel_size = postnet_config['kernel_size']
        padding = (kernel_size - 1) // 2 # Ensure sequence length is preserved

        layers = []
        # Initial layer
        layers.append(nn.Conv1d(mel_bins, channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm1d(channels))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(0.5)) # Common practice to add dropout

        # Intermediate layers
        for _ in range(n_convs - 2):
            layers.append(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.5))

        # Final layer (no activation/batchnorm)
        layers.append(nn.Conv1d(channels, mel_bins, kernel_size=kernel_size, padding=padding))
        layers.append(nn.Dropout(0.5)) # Dropout before final output

        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Mel spectrogram of shape (Batch, MelBins, MelTimeSteps).
        Returns:
            Tensor: Residual to add to the input mel spectrogram. Shape (Batch, MelBins, MelTimeSteps).
        """
        return self.conv_stack(x)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++

class MinimalSVS(pl.LightningModule):
    """
    Minimalist Singing Voice Synthesis model using PyTorch Lightning.
    Takes phonemes, durations, and F0 as input to predict mel spectrograms.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # Saves config to hparams
        self.config = config
        model_config = config['model']
        train_config = config['train']
        self.use_postnet = 'postnet' in model_config and model_config['postnet'] is not None

        # --- Layers ---
        self.phoneme_embedding = nn.Embedding(
            model_config['vocab_size'],
            model_config['embedding_dim'],
            padding_idx=0 # Assuming 0 is the padding index for phonemes
        )

        # RNN layer (LSTM)
        # Input features = embedding_dim + f0_dim
        rnn_input_dim = model_config['embedding_dim'] + model_config['f0_dim']
        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=model_config['rnn_hidden_dim'],
            num_layers=model_config['rnn_layers'],
            batch_first=True, # Crucial for (Batch, Seq, Features) input
            bidirectional=True # Make LSTM bidirectional
        )

        # Output projection layer
        # Input dimension needs to be doubled for bidirectional LSTM
        self.output_linear = nn.Linear(
            model_config['rnn_hidden_dim'] * 2, # Doubled input size
            model_config['mel_bins']
        )

        # Optional Post-Net
        if self.use_postnet:
            self.postnet = PostNet(config)
        else:
            self.postnet = None

        # --- Loss Function ---
        self.loss_fn = nn.L1Loss(reduction='none') # Get element-wise loss for masking

        # --- Training Params ---
        self.learning_rate = train_config['learning_rate']
        self.log_spec_every_n = train_config.get('log_spectrogram_every_n_val_steps', 50) # Default if not in config

    def _expand_inputs(self, phoneme_embeddings, durations):
        """
        Expands phoneme embeddings based on their durations to match frame-level length.

        Args:
            phoneme_embeddings (Tensor): Embeddings of shape (Batch, PhonemeSeqLen, EmbDim).
            durations (Tensor): Durations of shape (Batch, PhonemeSeqLen).

        Returns:
            Tensor: Expanded embeddings of shape (Batch, MelTimeSteps, EmbDim).
        """
        # Ensure durations are on the correct device and are integer type for repeat_interleave
        durations = durations.long().to(phoneme_embeddings.device)
        # Clamp durations to be at least 1 to avoid issues with zero duration
        durations = torch.clamp(durations, min=1)

        # Calculate total length for each item in the batch for validation
        total_lengths = torch.sum(durations, dim=1)
        max_len = torch.max(total_lengths)

        expanded_embeddings_list = []
        for i in range(phoneme_embeddings.size(0)): # Iterate through batch
            # Get non-padded durations and corresponding embeddings for this sample
            # Assuming padding value for duration is 0. Find first 0 duration.
            # Note: This assumes durations are padded with 0s. If not, adjust logic.
            valid_duration_indices = torch.nonzero(durations[i], as_tuple=True)[0]
            if len(valid_duration_indices) == 0: # Handle case with no valid durations
                 # Create a zero tensor of expected shape if no valid durations
                 # This might indicate an issue upstream or an empty sample
                 print(f"Warning: Sample {i} has no valid durations.")
                 # Use max_len calculated across batch, or a default if max_len is 0
                 expected_len = max_len if max_len > 0 else 1
                 expanded = torch.zeros((expected_len, phoneme_embeddings.size(2)),
                                        device=phoneme_embeddings.device,
                                        dtype=phoneme_embeddings.dtype)

            else:
                valid_durations = durations[i][valid_duration_indices]
                valid_embeddings = phoneme_embeddings[i][valid_duration_indices]

                # Use repeat_interleave
                # Input: (NumValidPhonemes, EmbDim), Repeats: (NumValidPhonemes)
                expanded = torch.repeat_interleave(valid_embeddings, valid_durations, dim=0)

                # Pad if necessary to match max_len within the batch
                current_len = expanded.size(0)
                if current_len < max_len:
                    padding_size = max_len - current_len
                    padding = torch.zeros((padding_size, expanded.size(1)),
                                          device=expanded.device, dtype=expanded.dtype)
                    expanded = torch.cat((expanded, padding), dim=0)
                elif current_len > max_len: # Should not happen if max_len is correct, but truncate just in case
                    expanded = expanded[:max_len, :]

            expanded_embeddings_list.append(expanded)

        # Stack the list of tensors into a single batch tensor
        batch_expanded_embeddings = torch.stack(expanded_embeddings_list, dim=0)
        # Expected shape: (Batch, MaxMelTimeSteps, EmbDim)
        return batch_expanded_embeddings, total_lengths

    def create_sequence_mask(self, lengths, max_len=None):
        """
        Creates a sequence mask from lengths.
        Args:
            lengths (Tensor): Long tensor of sequence lengths (Batch,).
            max_len (int, optional): Maximum sequence length. If None, uses max(lengths).
        Returns:
            Tensor: Boolean mask (Batch, MaxLen). True for valid positions, False for padding.
        """
        if max_len is None:
            max_len = lengths.max().item()
        # Create range tensor (0, 1, ..., max_len-1)
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0) # Shape (1, MaxLen)
        # Compare lengths (Batch, 1) with range (1, MaxLen) -> broadcasts to (Batch, MaxLen)
        mask = idx < lengths.unsqueeze(1)
        return mask


    def forward(self, phonemes, durations, f0):
        """
        Forward pass of the model.

        Args:
            phonemes (Tensor): Phoneme IDs (Batch, PhonemeSeqLen).
            durations (Tensor): Durations (Batch, PhonemeSeqLen).
            f0 (Tensor): F0 sequence (Batch, 1, MelTimeSteps) or (Batch, MelTimeSteps).

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If PostNet is used: (predicted_mel_pre, predicted_mel_post)
                - If PostNet is not used: predicted_mel_pre
                Both tensors have shape (Batch, MelBins, MelTimeSteps).
        """
        # 1. Embed Phonemes
        phoneme_embeddings = self.phoneme_embedding(phonemes) # (Batch, PhonemeSeqLen, EmbDim)

        # 2. Expand Embeddings based on Durations
        expanded_embeddings, _ = self._expand_inputs(phoneme_embeddings, durations)
        # expanded_embeddings shape: (Batch, MelTimeSteps, EmbDim)
        mel_time_steps = expanded_embeddings.size(1)

        # 3. Prepare F0
        # Ensure f0 has shape (Batch, MelTimeSteps, 1) for concatenation
        if f0.dim() == 2: # Input is (Batch, MelTimeSteps)
            f0 = f0.unsqueeze(-1) # -> (Batch, MelTimeSteps, 1)
        elif f0.dim() == 3 and f0.size(1) == 1: # Input is (Batch, 1, MelTimeSteps)
            f0 = f0.permute(0, 2, 1) # -> (Batch, MelTimeSteps, 1)
        # else: Shape is already (Batch, MelTimeSteps, 1) or incorrect

        # Ensure F0 matches the time steps of expanded embeddings
        # This might involve padding or truncating F0 if its length differs
        # from the sum of durations. The collate_fn should ideally handle this,
        # but we add a check/adjustment here for robustness.
        f0_time_steps = f0.size(1)
        if f0_time_steps < mel_time_steps:
            padding_size = mel_time_steps - f0_time_steps
            # Pad F0 at the end (time dimension)
            f0_padding = torch.zeros((f0.size(0), padding_size, f0.size(2)), device=f0.device, dtype=f0.dtype)
            f0 = torch.cat((f0, f0_padding), dim=1)
        elif f0_time_steps > mel_time_steps:
            f0 = f0[:, :mel_time_steps, :] # Truncate F0

        # 4. Concatenate Expanded Embeddings and F0
        # Input shape expected by LSTM: (Batch, SeqLen, InputDim)
        rnn_input = torch.cat((expanded_embeddings, f0), dim=2) # (Batch, MelTimeSteps, EmbDim + F0Dim)

        # 5. Pass through RNN
        rnn_output, _ = self.rnn(rnn_input) # (Batch, MelTimeSteps, RnnHiddenDim)

        # 6. Project to Mel Bins
        predicted_mels = self.output_linear(rnn_output) # (Batch, MelTimeSteps, MelBins)

        # 7. Reshape to (Batch, MelBins, MelTimeSteps) - Standard mel format
        predicted_mels_pre = predicted_mels.permute(0, 2, 1) # (Batch, MelBins, MelTimeSteps)

        # 8. Apply Post-Net if configured
        if self.postnet:
            residual = self.postnet(predicted_mels_pre)
            predicted_mels_post = predicted_mels_pre + residual
            return predicted_mels_pre, predicted_mels_post
        else:
            # Return pre-net prediction twice to maintain consistent output structure
            # Or adjust training/validation steps to handle single output
            return predicted_mels_pre, predicted_mels_pre # Returning pre-net result as post-net result when disabled

    def training_step(self, batch, batch_idx):
        phonemes = batch['phoneme']
        durations = batch['duration']
        f0 = batch['f0']
        target_mel = batch['mel']       # Shape (Batch, MelBins, MaxTimeSteps)
        mel_lengths = batch['mel_lengths'] # Shape (Batch,) - Added in collate_fn

        # Ensure target_mel has the correct shape (Batch, MelBins, Time) - Keep this check
        if target_mel.dim() == 3 and target_mel.size(1) != self.config['model']['mel_bins']:
             if target_mel.size(2) == self.config['model']['mel_bins']:
                 target_mel = target_mel.permute(0, 2, 1)
             else:
                 raise ValueError(f"Unexpected target_mel shape: {target_mel.shape}. Expected MelBins={self.config['model']['mel_bins']}")

        predicted_mel_pre, predicted_mel_post = self(phonemes, durations, f0)

        # --- Loss Calculation with Masking ---
        max_len = target_mel.size(2) # Max length in this batch

        # Create mask based on mel_lengths
        # Mask shape: (Batch, MaxLen) -> (Batch, 1, MaxLen) for broadcasting with (Batch, MelBins, MaxLen)
        mask = self.create_sequence_mask(mel_lengths, max_len).unsqueeze(1).to(target_mel.device)
        # Calculate sum of mask elements for averaging loss correctly
        # Ensure mask_sum is not zero to avoid division by zero
        mask_sum = mask.sum().clamp(min=1e-9) # Total number of valid mel bins in the batch

        # --- Pre-Net Loss ---
        # Ensure prediction has same length as target (model should ideally output correct length based on durations,
        # but padding/truncating prediction might be needed if model output length differs from target max_len)
        pred_len_pre = predicted_mel_pre.size(2)
        if pred_len_pre != max_len:
             # Pad or truncate predicted_mel_pre to match target max_len
             if pred_len_pre < max_len:
                  padding_size = max_len - pred_len_pre
                  predicted_mel_pre = F.pad(predicted_mel_pre, (0, padding_size)) # Pad last dim
             else: # pred_len_pre > max_len
                  predicted_mel_pre = predicted_mel_pre[:, :, :max_len] # Truncate last dim

        loss_pre_elementwise = self.loss_fn(predicted_mel_pre, target_mel) # (Batch, MelBins, MaxLen)
        loss_pre_masked = loss_pre_elementwise * mask
        loss_pre = loss_pre_masked.sum() / mask_sum # Average over valid bins only

        self.log('train_loss_pre', loss_pre, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))

        # --- Post-Net Loss (if applicable) ---
        if self.postnet:
            # Align predicted_mel_post length similar to pre-net prediction
            pred_len_post = predicted_mel_post.size(2)
            if pred_len_post != max_len:
                 if pred_len_post < max_len:
                      padding_size = max_len - pred_len_post
                      predicted_mel_post = F.pad(predicted_mel_post, (0, padding_size))
                 else: # pred_len_post > max_len
                      predicted_mel_post = predicted_mel_post[:, :, :max_len]

            loss_post_elementwise = self.loss_fn(predicted_mel_post, target_mel) # (Batch, MelBins, MaxLen)
            loss_post_masked = loss_post_elementwise * mask
            loss_post = loss_post_masked.sum() / mask_sum # Average over valid bins only

            self.log('train_loss_post', loss_post, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
            total_loss = loss_pre + loss_post
        else:
            total_loss = loss_pre # Only pre-net loss if post-net is disabled

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        phonemes = batch['phoneme']
        durations = batch['duration']
        f0 = batch['f0']
        target_mel = batch['mel']       # Shape (Batch, MelBins, MaxTimeSteps)
        mel_lengths = batch['mel_lengths'] # Shape (Batch,)

        # Ensure target_mel has the correct shape (Batch, MelBins, Time) - Keep this check
        if target_mel.dim() == 3 and target_mel.size(1) != self.config['model']['mel_bins']:
             if target_mel.size(2) == self.config['model']['mel_bins']:
                 target_mel = target_mel.permute(0, 2, 1)
             else:
                 raise ValueError(f"Unexpected target_mel shape: {target_mel.shape}. Expected MelBins={self.config['model']['mel_bins']}")

        predicted_mel_pre, predicted_mel_post = self(phonemes, durations, f0)

        # --- Loss Calculation with Masking ---
        max_len = target_mel.size(2) # Max length in this batch

        # Create mask based on mel_lengths
        mask = self.create_sequence_mask(mel_lengths, max_len).unsqueeze(1).to(target_mel.device) # (Batch, 1, MaxLen)
        # Calculate sum of mask elements for averaging loss correctly
        # Ensure mask_sum is not zero to avoid division by zero
        mask_sum = mask.sum().clamp(min=1e-9) # Total number of valid mel bins in the batch

        # --- Pre-Net Loss ---
        # Align predicted_mel_pre length to target max_len
        pred_len_pre = predicted_mel_pre.size(2)
        if pred_len_pre != max_len:
             if pred_len_pre < max_len:
                  padding_size = max_len - pred_len_pre
                  predicted_mel_pre = F.pad(predicted_mel_pre, (0, padding_size))
             else: # pred_len_pre > max_len
                  predicted_mel_pre = predicted_mel_pre[:, :, :max_len]

        loss_pre_elementwise = self.loss_fn(predicted_mel_pre, target_mel)
        loss_pre_masked = loss_pre_elementwise * mask
        loss_pre = loss_pre_masked.sum() / mask_sum

        self.log('val_loss_pre', loss_pre, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))

        # --- Post-Net Loss (if applicable) ---
        mel_for_plotting = predicted_mel_pre # Default to pre-net prediction for plotting
        if self.postnet:
            # Align predicted_mel_post length to target max_len
            pred_len_post = predicted_mel_post.size(2)
            if pred_len_post != max_len:
                 if pred_len_post < max_len:
                      padding_size = max_len - pred_len_post
                      predicted_mel_post = F.pad(predicted_mel_post, (0, padding_size))
                 else: # pred_len_post > max_len
                      predicted_mel_post = predicted_mel_post[:, :, :max_len]

            loss_post_elementwise = self.loss_fn(predicted_mel_post, target_mel)
            loss_post_masked = loss_post_elementwise * mask
            loss_post = loss_post_masked.sum() / mask_sum

            self.log('val_loss_post', loss_post, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
            total_loss = loss_pre + loss_post
            mel_for_plotting = predicted_mel_post # Use post-net prediction for plotting
        else:
            total_loss = loss_pre # Only pre-net loss if post-net is disabled

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0))

        # --- Log Spectrogram Comparison ---
        if self.logger and self.log_spec_every_n > 0 and batch_idx % self.log_spec_every_n == 0:
            try:
                # Select the first sample from the batch
                # Get the actual length of the first sample
                first_sample_len = mel_lengths[0].item()

                # Plot only the valid part of the spectrograms
                gt_mel_sample = target_mel[0, :, :first_sample_len].cpu().detach()    # Shape: (MelBins, ActualTime)
                pred_mel_sample = mel_for_plotting[0, :, :first_sample_len].cpu().detach() # Shape: (MelBins, ActualTime)

                # Basic check for valid 2D spectrogram shapes before plotting
                if len(gt_mel_sample.shape) == 2 and len(pred_mel_sample.shape) == 2 and gt_mel_sample.shape[0] == pred_mel_sample.shape[0] and gt_mel_sample.shape[1] > 0:
                    # Pass vmin/vmax=None for now, allowing auto-scaling
                    # TODO: Consider reading vmin/vmax from config if defined there later
                    fig = plot_spectrograms_to_figure(
                        gt_mel_sample,
                        pred_mel_sample,
                        title=f"Epoch {self.current_epoch} Batch {batch_idx} - Spec Comp (PostNet: {self.use_postnet})",
                        vmin=None,
                        vmax=None
                    )
                    self.logger.experiment.add_figure(
                        f"Validation_Images/Spectrogram_Comparison_Batch_{batch_idx}",
                        fig,
                        global_step=self.global_step
                    )
                    plt.close(fig)
                else:
                     print(f"Warning: Skipping spectrogram plot for E{self.current_epoch} B{batch_idx} due to shape mismatch or zero length. GT: {gt_mel_sample.shape}, Pred: {pred_mel_sample.shape}")
                     self.log(f"Warning/Spectrogram_Shape_Mismatch_B{batch_idx}", 1.0, on_step=True, on_epoch=False)

            except IndexError:
                 print(f"Warning: Skipping spectrogram plot for E{self.current_epoch} B{batch_idx} due to IndexError (likely batch size 0 or issue accessing index 0).")
                 self.log(f"Warning/Spectrogram_IndexError_B{batch_idx}", 1.0, on_step=True, on_epoch=False)
            except Exception as e:
                print(f"Warning: Error generating/logging spectrogram plot for E{self.current_epoch} B{batch_idx}: {e}")
                self.log(f"Warning/Spectrogram_Plot_Error_B{batch_idx}", 1.0, on_step=True, on_epoch=False)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Use ReduceLROnPlateau scheduler based on validation loss
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val_loss', # Metric to monitor
            'interval': 'epoch',   # Check scheduler condition every epoch
            'frequency': 1         # Check every validation epoch
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}