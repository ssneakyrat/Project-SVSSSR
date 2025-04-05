# model.py - SVS GAN Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_spectrograms_to_figure

# +++++++++++++++++ Helper Functions +++++++++++++++++
def _expand_inputs(phoneme_embeddings, durations):
    """
    Expands phoneme embeddings based on their durations to match frame-level length.
    (Adapted from previous model)
    """
    durations = durations.long().to(phoneme_embeddings.device)
    durations = torch.clamp(durations, min=1)
    total_lengths = torch.sum(durations, dim=1)
    max_len = torch.max(total_lengths)
    if max_len == 0: # Handle edge case where all durations might be zero in a batch
        max_len = 1

    expanded_embeddings_list = []
    for i in range(phoneme_embeddings.size(0)):
        valid_duration_indices = torch.nonzero(durations[i], as_tuple=True)[0]
        if len(valid_duration_indices) == 0:
            # Use max_len calculated across batch
            expanded = torch.zeros((max_len, phoneme_embeddings.size(2)),
                                     device=phoneme_embeddings.device,
                                     dtype=phoneme_embeddings.dtype)
        else:
            valid_durations = durations[i][valid_duration_indices]
            valid_embeddings = phoneme_embeddings[i][valid_duration_indices]
            expanded = torch.repeat_interleave(valid_embeddings, valid_durations, dim=0)

            current_len = expanded.size(0)
            if current_len < max_len:
                padding_size = max_len - current_len
                padding = torch.zeros((padding_size, expanded.size(1)),
                                      device=expanded.device, dtype=expanded.dtype)
                expanded = torch.cat((expanded, padding), dim=0)
            elif current_len > max_len:
                expanded = expanded[:max_len, :]

        expanded_embeddings_list.append(expanded)

    batch_expanded_embeddings = torch.stack(expanded_embeddings_list, dim=0)
    return batch_expanded_embeddings, total_lengths

def create_sequence_mask(lengths, max_len=None):
    """
    Creates a sequence mask from lengths.
    (Adapted from previous model)
    """
    if max_len is None:
        max_len = lengths.max().item()
        if max_len == 0: # Handle edge case
             max_len = 1
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask

# +++++++++++++++++ PostNet Definition +++++++++++++++++
# Reusing the PostNet from the previous model
class PostNet(nn.Module):
    """Post-Net for refining predicted Mel spectrograms."""
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        postnet_config = model_config.get('postnet', {}) # Use .get for flexibility
        mel_bins = model_config['mel_bins']
        # Provide defaults if not in config
        n_convs = postnet_config.get('n_convs', 5)
        channels = postnet_config.get('channels', 512)
        kernel_size = postnet_config.get('kernel_size', 5)
        dropout = postnet_config.get('dropout', 0.5)
        padding = (kernel_size - 1) // 2

        layers = []
        layers.append(nn.Conv1d(mel_bins, channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm1d(channels))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout))

        for _ in range(n_convs - 2):
            layers.append(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(channels, mel_bins, kernel_size=kernel_size, padding=padding))
        layers.append(nn.Dropout(dropout)) # Dropout before final output

        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        """ x: (Batch, MelBins, MelTimeSteps) """
        return self.conv_stack(x)

# +++++++++++++++++ Generator Definition +++++++++++++++++
class GeneratorSVS(nn.Module):
    """
    Generator network for SVS GAN.
    Takes phonemes, durations, and F0 as input to predict mel spectrograms.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        gen_config = model_config.get('generator', {}) # Generator specific config

        # --- Layers ---
        self.phoneme_embedding = nn.Embedding(
            model_config['vocab_size'],
            gen_config.get('embedding_dim', 256), # Example default
            padding_idx=0
        )

        # RNN layer (LSTM or GRU)
        rnn_input_dim = gen_config.get('embedding_dim', 256) + model_config.get('f0_dim', 1)
        self.rnn = nn.LSTM( # Or nn.GRU
            input_size=rnn_input_dim,
            hidden_size=gen_config.get('rnn_hidden_dim', 512), # Example default
            num_layers=gen_config.get('rnn_layers', 2),       # Example default
            batch_first=True,
            bidirectional=gen_config.get('bidirectional', True) # Example default
        )

        # Output projection layer
        rnn_output_dim = gen_config.get('rnn_hidden_dim', 512) * (2 if gen_config.get('bidirectional', True) else 1)
        self.output_linear = nn.Linear(
            rnn_output_dim,
            model_config['mel_bins']
        )

        # Optional Post-Net
        self.use_postnet = 'postnet' in model_config and model_config['postnet'] is not None
        if self.use_postnet:
            self.postnet = PostNet(config)
        else:
            self.postnet = None

    def forward(self, phonemes, durations, f0):
        """
        Args:
            phonemes (Tensor): Phoneme IDs (Batch, PhonemeSeqLen).
            durations (Tensor): Durations (Batch, PhonemeSeqLen).
            f0 (Tensor): F0 sequence (Batch, 1, MelTimeSteps) or (Batch, MelTimeSteps).

        Returns:
            Tuple[Tensor, Tensor]: predicted_mel_pre, predicted_mel_post
                                   (Batch, MelBins, MelTimeSteps)
        """
        # 1. Embed Phonemes
        phoneme_embeddings = self.phoneme_embedding(phonemes) # (B, PhSeq, Emb)

        # 2. Expand Embeddings based on Durations
        expanded_embeddings, _ = _expand_inputs(phoneme_embeddings, durations) # (B, MelT, Emb)
        mel_time_steps = expanded_embeddings.size(1)

        # 3. Prepare F0
        if f0.dim() == 2: # (B, MelT) -> (B, MelT, 1)
            f0 = f0.unsqueeze(-1)
        elif f0.dim() == 3 and f0.size(1) == 1: # (B, 1, MelT) -> (B, MelT, 1)
            f0 = f0.permute(0, 2, 1)

        # Ensure F0 matches time steps
        f0_time_steps = f0.size(1)
        if f0_time_steps < mel_time_steps:
            padding_size = mel_time_steps - f0_time_steps
            f0_padding = torch.zeros((f0.size(0), padding_size, f0.size(2)), device=f0.device, dtype=f0.dtype)
            f0 = torch.cat((f0, f0_padding), dim=1)
        elif f0_time_steps > mel_time_steps:
            f0 = f0[:, :mel_time_steps, :]

        # 4. Concatenate Expanded Embeddings and F0
        rnn_input = torch.cat((expanded_embeddings, f0), dim=2) # (B, MelT, Emb + F0Dim)

        # 5. Pass through RNN
        rnn_output, _ = self.rnn(rnn_input) # (B, MelT, RnnHidden*Directions)

        # 6. Project to Mel Bins
        predicted_mels = self.output_linear(rnn_output) # (B, MelT, MelBins)

        # 7. Reshape
        predicted_mels_pre = predicted_mels.permute(0, 2, 1) # (B, MelBins, MelT)

        # 8. Apply Post-Net
        if self.postnet:
            residual = self.postnet(predicted_mels_pre)
            predicted_mels_post = predicted_mels_pre + residual
        else:
            predicted_mels_post = predicted_mels_pre # Return pre-net if post-net disabled

        return predicted_mels_pre, predicted_mels_post


# +++++++++++++++++ Discriminator Definition +++++++++++++++++
class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Mel Spectrograms.
    Takes a Mel spectrogram (B, C=1, H=MelBins, W=MelTimeSteps) and outputs a patch map of real/fake predictions.
    """
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        disc_config = model_config.get('discriminator', {}) # Discriminator specific config
        mel_bins = model_config['mel_bins'] # Needed for input size calculation if using adaptive pooling/flatten

        # Configurable parameters with defaults
        in_channels = disc_config.get('in_channels', 1) # Input is single-channel Mel spec
        n_layers = disc_config.get('n_layers', 4)
        base_channels = disc_config.get('base_channels', 64)
        kernel_size = disc_config.get('kernel_size', (3, 9)) # Often asymmetric for spectrograms (Freq, Time)
        stride = disc_config.get('stride', (1, 2))
        padding = disc_config.get('padding', (1, 4)) # Adjust padding based on kernel/stride
        norm_layer_type = disc_config.get('norm_layer', 'batch') # 'batch' or 'instance'

        if norm_layer_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError("Unsupported norm_layer type. Choose 'batch' or 'instance'.")


        layers = []
        # Initial Conv layer (no norm)
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        channels = base_channels
        for i in range(1, n_layers):
            prev_channels = channels
            channels = min(base_channels * (2**i), 512) # Increase channels, cap at 512
            layers.append(nn.Conv2d(prev_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(norm_layer(channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final output layer (1 channel prediction map)
        layers.append(nn.Conv2d(channels, 1, kernel_size=kernel_size, stride=1, padding=padding))
        # No sigmoid here, use BCEWithLogitsLoss later

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input Mel spectrogram (Batch, 1, MelBins, MelTimeSteps).
                        Needs unsqueezing dim 1 if input is (B, MelBins, MelT).
        Returns:
            Tensor: Patch map of predictions (Batch, 1, H', W').
        """
        if x.dim() == 3: # Add channel dimension if missing
             x = x.unsqueeze(1) # (B, MelBins, MelT) -> (B, 1, MelBins, MelT)
        return self.model(x)


# +++++++++++++++++ Lightning Module for GAN Training +++++++++++++++++
class SVSGAN(pl.LightningModule):
    """ PyTorch Lightning Module for training the SVS GAN """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # Saves config to hparams
        self.config = config
        model_config = config['model']
        train_config = config['train']

        self.generator = GeneratorSVS(config)
        self.discriminator = PatchDiscriminator(config)

        # --- Loss Functions ---
        # Adversarial loss (use BCEWithLogitsLoss for stability with PatchGAN)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        # Reconstruction/Feature Matching loss
        self.recon_loss = nn.L1Loss(reduction='mean') # Mean over batch and features

        # --- Loss Weights ---
        # Lambda for reconstruction loss (how much to weight L1 vs adversarial)
        self.lambda_recon = train_config.get('lambda_recon', 100) # Common value, adjust as needed

        # --- Training Params ---
        self.learning_rate_g = train_config.get('learning_rate_g', 1e-4)
        self.learning_rate_d = train_config.get('learning_rate_d', 4e-4) # Often D learns faster
        self.b1 = train_config.get('adam_beta1', 0.5) # Betas common for GANs
        self.b2 = train_config.get('adam_beta2', 0.999)
        self.log_spec_every_n = train_config.get('log_spectrogram_every_n_val_steps', 50)

        # Automatic optimization will be turned off
        self.automatic_optimization = False

    def forward(self, phonemes, durations, f0):
        """ Generates Mel spectrograms using the generator """
        # During inference/validation, we only need the generator's output
        _, generated_mel_post = self.generator(phonemes, durations, f0)
        return generated_mel_post # Return the refined prediction

    def _get_aligned_predictions(self, predicted_mel, target_mel):
        """ Helper to align prediction length to target length """
        pred_len = predicted_mel.size(2)
        target_len = target_mel.size(2)
        if pred_len != target_len:
            if pred_len < target_len:
                padding_size = target_len - pred_len
                predicted_mel = F.pad(predicted_mel, (0, padding_size))
            else:
                predicted_mel = predicted_mel[:, :, :target_len]
        return predicted_mel

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        phonemes = batch['phoneme']
        durations = batch['duration']
        f0 = batch['f0']
        real_mel = batch['mel']       # Shape (Batch, MelBins, MaxTimeSteps)
        mel_lengths = batch['mel_lengths'] # Shape (Batch,)

        # Ensure real_mel has correct shape
        if real_mel.dim() == 3 and real_mel.size(1) != self.config['model']['mel_bins']:
             if real_mel.size(2) == self.config['model']['mel_bins']:
                 real_mel = real_mel.permute(0, 2, 1)
             else:
                 raise ValueError(f"Unexpected target_mel shape: {real_mel.shape}")

        # Generate fake mel spectrograms
        # We need both pre and post for potential losses, but typically use post for D
        fake_mel_pre, fake_mel_post = self.generator(phonemes, durations, f0)

        # Align fake_mel_post length with real_mel length for loss calculation
        fake_mel_post_aligned = self._get_aligned_predictions(fake_mel_post, real_mel)
        # Detach fake mel when training discriminator
        fake_mel_post_detached = fake_mel_post_aligned.detach()

        # --- Train Discriminator ---
        # Requires grads for D parameters
        self.toggle_optimizer(optimizer_d)

        # Real loss
        pred_real = self.discriminator(real_mel)
        # Target is matrix of 1s for real images
        target_real = torch.ones_like(pred_real)
        loss_d_real = self.adversarial_loss(pred_real, target_real)

        # Fake loss
        pred_fake = self.discriminator(fake_mel_post_detached)
        # Target is matrix of 0s for fake images
        target_fake = torch.zeros_like(pred_fake)
        loss_d_fake = self.adversarial_loss(pred_fake, target_fake)

        # Combined discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5 # Average

        # Optimize Discriminator
        optimizer_d.zero_grad()
        self.manual_backward(loss_d)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # Log Discriminator losses
        self.log('train_loss_d_real', loss_d_real, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_loss_d_fake', loss_d_fake, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_loss_d', loss_d, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0))


        # --- Train Generator ---
        # Requires grads for G parameters
        self.toggle_optimizer(optimizer_g)

        # Adversarial Loss (Generator wants Discriminator to predict 1 for fake images)
        # Use the non-detached fake mel here
        pred_fake_for_g = self.discriminator(fake_mel_post_aligned)
        target_real_for_g = torch.ones_like(pred_fake_for_g) # Target is 1s
        loss_g_adv = self.adversarial_loss(pred_fake_for_g, target_real_for_g)

        # Reconstruction Loss (L1 between generated and real mel)
        # Use mask to compute loss only on valid parts
        max_len = real_mel.size(2)
        mask = create_sequence_mask(mel_lengths, max_len).unsqueeze(1).to(real_mel.device) # (B, 1, T)
        mask_sum = mask.sum().clamp(min=1e-9)

        # Align pre-net prediction as well if needed for loss
        fake_mel_pre_aligned = self._get_aligned_predictions(fake_mel_pre, real_mel)

        # L1 loss on post-net output
        loss_g_recon_post = (self.recon_loss(fake_mel_post_aligned * mask, real_mel * mask) * mask.numel()) / mask_sum
        # Optional: L1 loss on pre-net output
        loss_g_recon_pre = (self.recon_loss(fake_mel_pre_aligned * mask, real_mel * mask) * mask.numel()) / mask_sum

        # Total Generator Loss
        # Combine adversarial and reconstruction losses
        # You might only use post-net recon loss or both pre+post
        loss_g = loss_g_adv + (loss_g_recon_post * self.lambda_recon) # + (loss_g_recon_pre * self.lambda_recon) # Optionally add pre-net loss

        # Optimize Generator
        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # Log Generator losses
        self.log('train_loss_g_adv', loss_g_adv, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_loss_g_recon_post', loss_g_recon_post, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_loss_g_recon_pre', loss_g_recon_pre, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('train_loss_g', loss_g, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0))


    def validation_step(self, batch, batch_idx):
        # Validation focuses on Generator's performance (reconstruction)
        # and potentially logs generated images
        phonemes = batch['phoneme']
        durations = batch['duration']
        f0 = batch['f0']
        target_mel = batch['mel']       # Shape (Batch, MelBins, MaxTimeSteps)
        mel_lengths = batch['mel_lengths'] # Shape (Batch,)

        if target_mel.dim() == 3 and target_mel.size(1) != self.config['model']['mel_bins']:
             if target_mel.size(2) == self.config['model']['mel_bins']:
                 target_mel = target_mel.permute(0, 2, 1)
             else:
                 raise ValueError(f"Unexpected target_mel shape: {target_mel.shape}")

        predicted_mel_pre, predicted_mel_post = self.generator(phonemes, durations, f0)

        # Align lengths
        predicted_mel_pre_aligned = self._get_aligned_predictions(predicted_mel_pre, target_mel)
        predicted_mel_post_aligned = self._get_aligned_predictions(predicted_mel_post, target_mel)

        # --- Loss Calculation with Masking ---
        max_len = target_mel.size(2)
        mask = create_sequence_mask(mel_lengths, max_len).unsqueeze(1).to(target_mel.device)
        mask_sum = mask.sum().clamp(min=1e-9)

        # --- Reconstruction Losses (similar to Generator's recon loss) ---
        loss_pre = (self.recon_loss(predicted_mel_pre_aligned * mask, target_mel * mask) * mask.numel()) / mask_sum
        loss_post = (self.recon_loss(predicted_mel_post_aligned * mask, target_mel * mask) * mask.numel()) / mask_sum

        self.log('val_loss_pre', loss_pre, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('val_loss_post', loss_post, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=phonemes.size(0))
        self.log('val_loss', loss_post, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=phonemes.size(0)) # Use post-net loss as main val metric

        # --- Log Spectrogram Comparison ---
        if self.logger and self.log_spec_every_n > 0 and batch_idx % self.log_spec_every_n == 0:
            try:
                first_sample_len = mel_lengths[0].item()
                if first_sample_len > 0: # Ensure there's something to plot
                    gt_mel_sample = target_mel[0, :, :first_sample_len].cpu().detach()
                    pred_mel_sample = predicted_mel_post_aligned[0, :, :first_sample_len].cpu().detach() # Use post-net

                    if len(gt_mel_sample.shape) == 2 and len(pred_mel_sample.shape) == 2 and gt_mel_sample.shape[0] == pred_mel_sample.shape[0]:
                        fig = plot_spectrograms_to_figure(
                            gt_mel_sample,
                            pred_mel_sample,
                            title=f"Val Epoch {self.current_epoch} Batch {batch_idx} - Spec Comp",
                            vmin=None, vmax=None # Auto-scale for now
                        )
                        self.logger.experiment.add_figure(
                            f"Validation_Images/Spectrogram_Comparison_Batch_{batch_idx}",
                            fig,
                            global_step=self.global_step
                        )
                        plt.close(fig)
                    else:
                         print(f"Warning: Skipping val spec plot B{batch_idx} due to shape mismatch. GT: {gt_mel_sample.shape}, Pred: {pred_mel_sample.shape}")
                         self.log(f"Warning/Val_Spec_Shape_Mismatch_B{batch_idx}", 1.0)
                else:
                    print(f"Warning: Skipping val spec plot B{batch_idx} due to zero length.")
                    self.log(f"Warning/Val_Spec_Zero_Length_B{batch_idx}", 1.0)

            except IndexError:
                 print(f"Warning: Skipping val spec plot B{batch_idx} due to IndexError.")
                 self.log(f"Warning/Val_Spec_IndexError_B{batch_idx}", 1.0)
            except Exception as e:
                print(f"Warning: Error generating/logging val spec plot B{batch_idx}: {e}")
                self.log(f"Warning/Val_Spec_Plot_Error_B{batch_idx}", 1.0)

        return {'val_loss': loss_post} # Return dict for potential callbacks


    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate_g,
            betas=(self.b1, self.b2)
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate_d,
            betas=(self.b1, self.b2)
        )

        # Optional: Learning rate schedulers
        # Example: ReduceLROnPlateau for the generator based on validation loss
        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val_loss', # Monitor generator's performance proxy
            'interval': 'epoch',
            'frequency': 1
        }
        # No scheduler for D typically, or a simpler one if needed

        # Return optimizers and schedulers
        # Note: Pytorch Lightning handles the manual optimization logic based on us returning two optimizers
        # when automatic_optimization = False
        return [optimizer_g, optimizer_d], [scheduler_g] # Return schedulers in a list if needed