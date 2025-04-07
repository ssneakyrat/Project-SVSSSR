# models/progressive_svs.py
import numpy as np

import logging # Import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision
import torchaudio # Add torchaudio for Mel Spectrogram calculation
import torchaudio.transforms as T # Add torchaudio transforms
from models.blocks import LowResModel, MidResUpsampler, HighResUpsampler
from utils.plotting import plot_spectrograms_to_figure, plot_waveforms_to_figure, plot_spectrograms_comparison_to_figure, plot_single_spectrogram_to_figure # Import the plotting utilities # Add the new function
# Imports will be handled inside __init__ now

logger = logging.getLogger(__name__) # Get logger instance


class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.f0_embed_dim = config['model']['f0_embed_dim']
        self.phone_embed_dim = config['model']['phone_embed_dim']
        self.midi_embed_dim = config['model']['midi_embed_dim']
        self.unvoiced_embed_dim = config['model']['unvoiced_embed_dim'] # Read from config

        # F0 encoding (continuous value)
        self.f0_projection = nn.Linear(1, self.f0_embed_dim)

        # Phone encoding (categorical)
        num_phones = len(config.get('phone_map', []))
        if num_phones == 0:
            num_phones = 100  # Default if not specified
        self.phone_embedding = nn.Embedding(num_phones, self.phone_embed_dim)

        # MIDI encoding (categorical)
        num_midi_notes = 128  # MIDI standard
        self.midi_embedding = nn.Embedding(num_midi_notes, self.midi_embed_dim)

        # Unvoiced flag encoding (continuous value 0.0 or 1.0)
        self.unvoiced_projection = nn.Linear(1, self.unvoiced_embed_dim)

    def forward(self, f0, phone_label, phone_duration, midi_label, unvoiced_flag):
        # f0 has shape (B, T, 1), get B and T correctly
        batch_size = f0.size(0)
        seq_len = f0.size(1)

        # Encode F0 (frame-level)
        f0_encoded = self.f0_projection(f0)  # [B, T, F0_dim]

        # Encode phone (frame-level)
        phone_encoded = self.phone_embedding(phone_label.long())  # [B, T, P_dim]

        # Encode MIDI (frame-level)
        midi_label_squeezed = midi_label.long()
        if midi_label_squeezed.dim() == 3 and midi_label_squeezed.size(2) == 1:
             midi_label_squeezed = midi_label_squeezed.squeeze(2) # Shape becomes (B, T)
        midi_encoded = self.midi_embedding(midi_label_squeezed)  # [B, T, M_dim]

        # Encode unvoiced flag (frame-level) separately
        if unvoiced_flag.dim() == 2:
            unvoiced_flag = unvoiced_flag.unsqueeze(-1)
        unvoiced_encoded = self.unvoiced_projection(unvoiced_flag) # [B, T, U_dim]

        # Return individual embeddings before concatenation/transposition
        # Shapes: [B, T, Dim]
        return f0_encoded, phone_encoded, midi_encoded, unvoiced_encoded

class ProgressiveSVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters('config')

        self.config = config
        self.current_stage = config['model'].get('current_stage', 1)

        # --- Perform imports locally within __init__ ---
        ProgressiveVocoder = None
        STFTLoss = None
        vocoder_imports_ok = False
        try:
            from models.progressive_vocoder import ProgressiveVocoder as VocoderCls
            from utils.losses import STFTLoss as LossCls
            ProgressiveVocoder = VocoderCls # Assign to local scope variable
            STFTLoss = LossCls             # Assign to local scope variable
            vocoder_imports_ok = True
            logger.info("DEBUG: Successfully imported ProgressiveVocoder and STFTLoss inside __init__.")
        except ImportError as e:
            logger.warning(f"Could not import ProgressiveVocoder or STFTLoss inside __init__. Stage 1 Vocoder training disabled. Error: {e}")
            # ProgressiveVocoder and STFTLoss remain None

        # Access the flag from the 'model' section, not 'train'
        self.use_stage1_vocoder = config['model'].get('stage1_use_vocoder', False) and vocoder_imports_ok
        logger.info(f"DEBUG: Checking config['model'].get('stage1_use_vocoder'): {config['model'].get('stage1_use_vocoder', 'Not Found')}") # Add deeper debug
        logger.info(f"DEBUG: vocoder_imports_ok: {vocoder_imports_ok}") # Add deeper debug
        logger.info(f"DEBUG: Final self.use_stage1_vocoder = {self.use_stage1_vocoder}") # Add deeper debug
        if config['model'].get('stage1_use_vocoder', False) and not vocoder_imports_ok: # Also correct path here
             logger.warning("Config requested stage1_use_vocoder=True, but imports failed during __init__. Disabling.")
        logger.info(f"DEBUG: self.use_stage1_vocoder initialized to: {self.use_stage1_vocoder}") # <-- Add DEBUG log

        # Feature encoder
        self.feature_encoder = FeatureEncoder(config)
        # Store individual embedding dims for later use
        self.f0_embed_dim = self.feature_encoder.f0_embed_dim
        self.phone_embed_dim = self.feature_encoder.phone_embed_dim
        self.midi_embed_dim = self.feature_encoder.midi_embed_dim
        self.unvoiced_embed_dim = self.feature_encoder.unvoiced_embed_dim

        # Progressive stages
        # Input dim for LowResModel (still uses concatenated main features)
        input_dim_main = self.f0_embed_dim + self.phone_embed_dim + self.midi_embed_dim
        # We stored individual dims earlier (self.f0_embed_dim, etc.)

        # Calculate frequency dimensions for each stage
        full_mel_bins = config['model']['mel_bins']
        low_res_freq_bins = int(full_mel_bins / config['model']['low_res_scale'])
        mid_res_freq_bins = int(full_mel_bins / config['model']['mid_res_scale'])

        # Stage 1: Low-Resolution Model
        # Determine the stride used in the first block of LowResModel for downsampling
        # Assuming stride=2 for the first block as per LowResModel implementation
        self.downsample_stride = 2 if len(config['model']['low_res_channels']) > 0 else 1

        self.low_res_model = LowResModel(
            input_dim=input_dim_main,
            channels_list=config['model']['low_res_channels'],
            output_dim=low_res_freq_bins,
            unvoiced_embed_dim=self.unvoiced_embed_dim, # Pass only unvoiced dim here
            config=config # Pass the config dictionary
        )

        # Stage 2: Mid-Resolution Upsampler
        self.mid_res_upsampler = MidResUpsampler(
            input_dim=low_res_freq_bins, # Freq dim from previous stage
            channels_list=config['model']['mid_res_channels'],
            output_dim=mid_res_freq_bins,
            f0_embed_dim=self.f0_embed_dim,
            phone_embed_dim=self.phone_embed_dim,
            midi_embed_dim=self.midi_embed_dim,
            unvoiced_embed_dim=self.unvoiced_embed_dim,
            downsample_stride=self.downsample_stride,
            config=config # Pass the config dictionary
        )

        # Stage 3: High-Resolution Upsampler
        self.high_res_upsampler = HighResUpsampler(
            input_dim=mid_res_freq_bins, # Freq dim from previous stage
            channels_list=config['model']['high_res_channels'], # Use updated channels
            output_dim=full_mel_bins,
            f0_embed_dim=self.f0_embed_dim,
            phone_embed_dim=self.phone_embed_dim,
            midi_embed_dim=self.midi_embed_dim,
            unvoiced_embed_dim=self.unvoiced_embed_dim,
            downsample_stride=self.downsample_stride, # Same stride as MidRes
            config=config # Pass the config dictionary
        )

        # --- Initialize Vocoder, Loss, and Resampler based on config ---
        self.progressive_vocoder = None
        self.audio_loss_fn = None
        self.stage1_resampler = None # Initialize resampler attribute
        self.stage1_target_sr = None # Initialize target SR attribute

        if self.use_stage1_vocoder:
            logger.info("Stage 1 Vocoder mode enabled. Initializing ProgressiveVocoder, STFTLoss, and Resampler.")
            # Initialize Vocoder
            if ProgressiveVocoder is not None:
                 self.progressive_vocoder = ProgressiveVocoder(config)
            else:
                 logger.error("ProgressiveVocoder class is None despite imports seeming available. Cannot initialize.")
                 self.use_stage1_vocoder = False # Disable if class is None

            # Initialize Loss Function (only if vocoder init succeeded)
            if self.use_stage1_vocoder and STFTLoss is not None:
                 try:
                     stft_loss_params = config['train'].get('stft_loss_params', {})
                     self.loss_fn = STFTLoss( # Use the locally imported STFTLoss
                         fft_sizes=stft_loss_params.get('fft_sizes', [1024, 2048, 512]),
                         hop_sizes=stft_loss_params.get('hop_sizes', [120, 240, 50]),
                         win_lengths=stft_loss_params.get('win_lengths', [600, 1200, 240]),
                         loss_sc_weight=stft_loss_params.get('loss_sc_weight', 1.0),
                         loss_mag_weight=stft_loss_params.get('loss_mag_weight', 1.0)
                     )
                     self.audio_loss_fn = self.loss_fn # Assign specifically for clarity later
                     logger.info("DEBUG: Successfully assigned STFTLoss to self.audio_loss_fn.")
                 except Exception as e:
                     logger.error(f"Error initializing STFTLoss: {e}", exc_info=True)
                     self.use_stage1_vocoder = False # Disable if STFTLoss init fails
                     self.loss_fn = nn.L1Loss(reduction='none') # Fallback
                     self.audio_loss_fn = None
            elif self.use_stage1_vocoder: # STFTLoss was None
                 logger.error("STFTLoss class is None even though imports seemed okay. Cannot initialize audio loss.")
                 self.use_stage1_vocoder = False # Disable if loss cannot be initialized
                 self.loss_fn = nn.L1Loss(reduction='none') # Fallback
                 self.audio_loss_fn = None

            # Initialize Resampler (only if vocoder and loss init succeeded)
            if self.use_stage1_vocoder:
                 try:
                     orig_sr = self.config['audio']['sample_rate']
                     # Safely access nested keys for divisor
                     vocoder_config = self.config['model'].get('progressive_vocoder', {})
                     sr_divisor = vocoder_config.get('v1_output_sr_divisor')

                     if sr_divisor is None or not isinstance(sr_divisor, (int, float)) or sr_divisor <= 0:
                         logger.error(f"Invalid or missing 'v1_output_sr_divisor' in config['model']['progressive_vocoder']. Found: {sr_divisor}. Disabling resampling.")
                         self.use_stage1_vocoder = False # Need divisor to resample
                     else:
                         target_sr = int(orig_sr / sr_divisor) # Ensure integer SR
                         if target_sr <= 0:
                              logger.error(f"Calculated target sample rate ({target_sr}) is invalid. Disabling resampling.")
                              self.use_stage1_vocoder = False
                         else:
                              self.stage1_resampler = T.Resample(orig_sr, target_sr) # Use positional arguments
                              self.stage1_target_sr = target_sr # Store the target SR
                              logger.info(f"Initialized Stage 1 target audio resampler: {orig_sr}Hz -> {target_sr}Hz")

                 except KeyError as e:
                      logger.error(f"Missing key required for resampler initialization: {e}. Disabling resampling.")
                      self.use_stage1_vocoder = False # Disable if config keys missing
                 except Exception as e:
                      logger.error(f"Error initializing Stage 1 resampler: {e}", exc_info=True)
                      self.use_stage1_vocoder = False # Disable on other errors

        # Fallback / Default Loss if Stage 1 Vocoder is not used or failed initialization
        if not self.use_stage1_vocoder:
            logger.info("Stage 1 Vocoder mode disabled or failed init. Using L1 Mel Loss for Stage 1 if applicable.")
            # Default Mel L1 loss (will be used if stage 1 runs without vocoder, or for stages 2/3)
            self.loss_fn = nn.L1Loss(reduction='none')
            self.audio_loss_fn = None # Ensure audio loss is None

        logger.info(f"DEBUG: End of __init__: self.audio_loss_fn is None: {self.audio_loss_fn is None}") # <-- Add DEBUG log
        logger.info(f"DEBUG: End of __init__: self.stage1_resampler is None: {self.stage1_resampler is None}")


    def forward(self, f0, phone_label, phone_duration, midi_label, unvoiced_flag):
        # Encode input features - Returns individual embeddings [B, T_orig, Dim]
        f0_enc, phone_enc, midi_enc, unvoiced_enc = self.feature_encoder(
            f0, phone_label, phone_duration, midi_label, unvoiced_flag
        )

        # --- Prepare inputs for Stage 1 ---
        # Concatenate main features (F0, phone, MIDI) [B, T_orig, F0+P+M_dim]
        main_features_cat = torch.cat([f0_enc, phone_enc, midi_enc], dim=2)
        # Transpose for Conv1D: [B, F0+P+M_dim, T_orig]
        main_features_s1 = main_features_cat.transpose(1, 2)
        # Transpose unvoiced embedding for Conv1D: [B, U_dim, T_orig]
        unvoiced_embedding_s1 = unvoiced_enc.transpose(1, 2)

        # --- Stage 1: Low-Resolution Model ---
        # Input: Transposed main features and unvoiced embedding
        # Output: Low-res spectrogram [B, T_downsampled, F_low]
        low_res_output = self.low_res_model(main_features_s1, unvoiced_embedding_s1)


        if self.current_stage == 1:
            if self.use_stage1_vocoder and self.progressive_vocoder is not None:
                # Pass low-res mel to vocoder
                # Input shape: [B, T_downsampled, F_low]
                logger.debug(f"Forward: Stage 1 Vocoder Input Shape: {low_res_output.shape}")
                predicted_audio = self.progressive_vocoder(low_res_output)
                logger.debug(f"Forward: Stage 1 Vocoder Output Shape: {predicted_audio.shape}")
                # Output shape: [B, T_audio]
                return predicted_audio
            else:
                # Return low-res mel if vocoder is not used
                return low_res_output

        # --- Stage 2: Mid-Resolution Upsampler ---
        # Input: Low-res spec and ORIGINAL embeddings
        mid_res_output = self.mid_res_upsampler(
            low_res_output, f0_enc, phone_enc, midi_enc, unvoiced_enc
        )

        if self.current_stage == 2:
            return mid_res_output

        # --- Stage 3: High-Resolution Upsampler ---
        # Input: Mid-res spec and ORIGINAL embeddings
        high_res_output = self.high_res_upsampler(
            mid_res_output, f0_enc, phone_enc, midi_enc, unvoiced_enc
        )


        return high_res_output

    def _compute_loss_and_target(self, batch):
        # Returns:
        # If using vocoder: loss_total, loss_sc, loss_mag, predicted_audio, target_audio
        # If not using vocoder: loss_mel, None, None, mel_pred, mel_target
        """
        Computes the loss for the current stage, handling target downsampling/resampling and masking.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                loss_total, loss_sc | None, loss_mag | None, prediction, target
            Returns dummy loss (1e9) and None for prediction/target on error.
        """
        mel_specs = batch['mel_spec']
        f0 = batch['f0']
        phone_label = batch['phone_label']
        phone_duration = batch['phone_duration']
        midi_label = batch['midi_label']
        lengths = batch['length'] # Frame lengths for mel
        voiced_mask = batch['voiced_mask']
        unvoiced_flag = batch['unvoiced_flag']
        target_audio = batch.get('audio_waveform') # Get target audio if available
        audio_lengths = batch.get('audio_length') # Get audio lengths if available

        # --- Stage 1 Vocoder Loss Calculation ---
        if self.current_stage == 1 and self.use_stage1_vocoder:
            if target_audio is None or self.audio_loss_fn is None or self.stage1_resampler is None:
                 logger.error("Cannot compute audio loss: Target audio, audio loss function, or resampler is missing.")
                 # Return dummy high loss and None for outputs
                 return torch.tensor(1e9, device=self.device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0), None, None

            # Forward pass to get predicted audio
            predicted_audio = self(f0, phone_label, phone_duration, midi_label, unvoiced_flag) # Shape: [B, T_audio_pred]

            # Ensure target_audio has the correct shape [B, T_audio_orig] before resampling
            if target_audio.dim() == 3 and target_audio.shape[1] == 1:
                 target_audio = target_audio.squeeze(1)
            elif target_audio.dim() != 2:
                 logger.error(f"Unexpected target_audio shape before resampling: {target_audio.shape}. Expected [B, T].")
                 return torch.tensor(1e9, device=self.device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0), None, None

            # --- Resample Target Audio for Stage 1 ---
            try:
                original_target_shape = target_audio.shape
                # Ensure target_audio is on the same device as the model
                target_audio = target_audio.to(self.device)
                target_audio_resampled = self.stage1_resampler(target_audio)
                logger.debug(f"Resampled target audio from {original_target_shape} to {target_audio_resampled.shape}")
            except Exception as e:
                logger.error(f"Error during target audio resampling: {e}", exc_info=True)
                # Return dummy high loss if resampling fails
                return torch.tensor(1e9, device=self.device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0), None, None
            # --- End Resampling ---

            # Trim predicted or resampled target audio to the minimum length for loss calculation
            # --- DEBUG: Log shapes before trimming ---
            logger.info(f"DEBUG: Before trimming - Predicted audio shape: {predicted_audio.shape}")
            logger.info(f"DEBUG: Before trimming - Resampled target audio shape: {target_audio_resampled.shape}")
            # --- End DEBUG ---
            try:
                min_len = min(predicted_audio.shape[1], target_audio_resampled.shape[1])
                predicted_audio_trimmed = predicted_audio[:, :min_len]
                target_audio_trimmed = target_audio_resampled[:, :min_len]
                logger.debug(f"Trimmed audio waveforms to min_len: {min_len}")
            except Exception as e:
                 logger.error(f"Error trimming audio waveforms: {e}. Pred shape: {predicted_audio.shape}, Target shape: {target_audio_resampled.shape}")
                 return torch.tensor(1e9, device=self.device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0), None, None

            # Calculate STFT loss using trimmed waveforms
            # Note: Pass original audio_lengths if STFTLoss needs them for masking based on original duration
            loss_sc, loss_mag, loss_total = self.audio_loss_fn(predicted_audio_trimmed, target_audio_trimmed, lengths=audio_lengths)

            # Return audio losses and the *trimmed* waveforms used for loss
            return loss_total, loss_sc, loss_mag, predicted_audio_trimmed, target_audio_trimmed

        # --- Mel Loss Calculation (Stages 2, 3 or Stage 1 without vocoder) ---
        else:
            # Forward pass (will return mel spectrogram)
            mel_pred = self(f0, phone_label, phone_duration, midi_label, unvoiced_flag)

            # Create masks for variable length based on ORIGINAL time dimension
            original_max_len = mel_specs.shape[1]
            original_mask = torch.arange(original_max_len, device=lengths.device).expand(len(lengths), original_max_len) < lengths.unsqueeze(1)
            # Calculate target time dimension based on model's downsampling
            target_time_dim = original_max_len // self.downsample_stride

            # Adjust target mel resolution based on current stage
            freq_weights_tensor = None
            if self.current_stage == 1: # Mel loss for stage 1 (vocoder disabled)
                scale_factor = 1/self.config['model']['low_res_scale']
                freq_dim = int(self.config['model']['mel_bins'] * scale_factor)
                b, t, c = mel_specs.shape
                mel_specs_permuted = mel_specs.permute(0, 2, 1) # B, C, T
                mel_specs_4d = mel_specs_permuted.unsqueeze(1) # B, 1, C, T
                # Interpolate both freq and time
                mel_target = F.interpolate(mel_specs_4d, size=(freq_dim, target_time_dim), mode='bilinear', align_corners=False).squeeze(1) # B, F_low, T_down
                mel_target = mel_target.permute(0, 2, 1) # B, T_down, F_low
                # Create mask for downsampled time dimension
                mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # B, 1, 1, T_orig
                downsampled_mask = F.interpolate(mask_float, size=(1, target_time_dim), mode='nearest').squeeze(1).squeeze(1) # B, T_down
                mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # B, T_down, F_low
            elif self.current_stage == 2:
                scale_factor = 1/self.config['model']['mid_res_scale']
                freq_dim = int(self.config['model']['mel_bins'] * scale_factor)
                b, t, c = mel_specs.shape
                mel_specs_permuted = mel_specs.permute(0, 2, 1) # B, C, T
                mel_specs_4d = mel_specs_permuted.unsqueeze(1) # B, 1, C, T
                # Interpolate both freq and time
                mel_target = F.interpolate(mel_specs_4d, size=(freq_dim, target_time_dim), mode='nearest').squeeze(1) # B, F_mid, T_down
                mel_target = mel_target.permute(0, 2, 1) # B, T_down, F_mid
                # Create mask for downsampled time dimension
                mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # B, 1, 1, T_orig
                downsampled_mask = F.interpolate(mask_float, size=(1, target_time_dim), mode='nearest').squeeze(1).squeeze(1) # B, T_down
                mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # B, T_down, F_mid
                # Frequency weighting for higher stages
                freq_weights = torch.linspace(1.0, 2.0, freq_dim, device=self.device)
                freq_weights_tensor = freq_weights.unsqueeze(0).unsqueeze(0) # 1, 1, F_mid
            else: # Stage 3
                freq_dim = self.config['model']['mel_bins'] # Full frequency bins
                b, t, c = mel_specs.shape
                mel_specs_permuted = mel_specs.permute(0, 2, 1) # B, C, T
                mel_specs_4d = mel_specs_permuted.unsqueeze(1) # B, 1, C, T
                # Interpolate only time dimension
                mel_target = F.interpolate(mel_specs_4d, size=(freq_dim, target_time_dim), mode='bilinear', align_corners=False).squeeze(1) # B, F_full, T_down
                mel_target = mel_target.permute(0, 2, 1) # B, T_down, F_full
                # Create mask for downsampled time dimension
                mask_float = original_mask.float().unsqueeze(1).unsqueeze(1) # B, 1, 1, T_orig
                downsampled_mask = F.interpolate(mask_float, size=(1, target_time_dim), mode='nearest').squeeze(1).squeeze(1) # B, T_down
                mask = downsampled_mask.unsqueeze(2).expand(-1, -1, freq_dim) # B, T_down, F_full
                # Frequency weighting for higher stages
                freq_weights = torch.linspace(1.0, 2.0, freq_dim, device=self.device)
                freq_weights_tensor = freq_weights.unsqueeze(0).unsqueeze(0) # 1, 1, F_full

            # Handle dimensionality issues if mel_pred shape is unexpected
            if mel_pred.dim() != mel_target.dim():
                 logger.warning(f"Shape mismatch detected before loss. Pred: {mel_pred.shape}, Target: {mel_target.shape}. Attempting correction.")
                 # Example correction (adjust as needed based on typical errors)
                 if mel_pred.dim() == 4 and mel_pred.size(1) == 1: # B, 1, T, F -> B, T, F
                     mel_pred = mel_pred.squeeze(1)
                 elif mel_pred.dim() == 3 and mel_target.dim() == 3 and mel_pred.shape[1] != mel_target.shape[1]: # B, T_pred, F vs B, T_target, F
                     logger.warning(f"Time dimension mismatch after forward pass. Pred_T: {mel_pred.shape[1]}, Target_T: {mel_target.shape[1]}. Trimming to min.")
                     min_t = min(mel_pred.shape[1], mel_target.shape[1])
                     mel_pred = mel_pred[:, :min_t, :]
                     mel_target = mel_target[:, :min_t, :]
                     mask = mask[:, :min_t, :] # Also trim mask

            # Ensure shapes match before loss calculation
            if mel_pred.shape != mel_target.shape:
                logger.error(f"Unresolvable shape mismatch for Mel loss. Pred: {mel_pred.shape}, Target: {mel_target.shape}. Skipping loss calculation.")
                return torch.tensor(1e9, device=self.device, requires_grad=True), None, None, None, None # Return dummy loss and None

            # Compute element-wise loss (using self.loss_fn, which is L1Loss here)
            loss_elementwise = self.loss_fn(mel_pred, mel_target)

            # Apply Weighted Loss for voiced/unvoiced frames
            unvoiced_weight = self.config['train'].get('unvoiced_weight', 1.0) # Get weight from config, default 1.0
            if voiced_mask.dim() == 3 and voiced_mask.shape[-1] == 1:
                 voiced_mask = voiced_mask.squeeze(-1) # B, T_orig
            # Downsample voiced mask to match target time dimension
            voiced_mask_float = voiced_mask.float().unsqueeze(1).unsqueeze(1) # B, 1, 1, T_orig
            downsampled_voiced_mask = F.interpolate(voiced_mask_float, size=(1, target_time_dim), mode='nearest').squeeze(1).squeeze(1) # B, T_down
            downsampled_voiced_mask = downsampled_voiced_mask > 0.5 # Convert back to bool
            # Create weights tensor (B, T_down, 1)
            loss_weights = torch.where(downsampled_voiced_mask.unsqueeze(-1), 1.0, unvoiced_weight)
            loss_weights = loss_weights.to(loss_elementwise.device) # Ensure same device

            # Apply length mask, loss weights, and frequency weights
            # Ensure mask has the correct shape (B, T_down, F)
            if mask.shape != loss_elementwise.shape:
                 logger.error(f"Mask shape {mask.shape} does not match loss shape {loss_elementwise.shape}. Skipping weighted loss.")
                 return torch.tensor(1e9, device=self.device, requires_grad=True), None, None, mel_pred, mel_target

            if freq_weights_tensor is not None:
                # Ensure freq_weights_tensor is broadcastable (1, 1, F)
                if freq_weights_tensor.shape != (1, 1, loss_elementwise.shape[2]):
                     logger.error(f"Freq weights shape {freq_weights_tensor.shape} not broadcastable to loss shape {loss_elementwise.shape}. Skipping freq weighting.")
                     freq_weights_tensor = torch.ones_like(freq_weights_tensor) # Use ones to disable effect

                weighted_loss = loss_elementwise * mask.float() * loss_weights * freq_weights_tensor
                total_weight_sum = (mask.float() * loss_weights * freq_weights_tensor).sum()
            else: # Stage 1 (Mel loss)
                weighted_loss = loss_elementwise * mask.float() * loss_weights
                total_weight_sum = (mask.float() * loss_weights).sum()

            loss_mel = weighted_loss.sum() / total_weight_sum.clamp(min=1e-8)

            # Return mel loss (with None for audio loss components) and mel spectrograms
            return loss_mel, None, None, mel_pred, mel_target


    def training_step(self, batch, batch_idx):
        # Unpack potentially different return values based on mode
        # Vocoder mode: loss_total, loss_sc, loss_mag, predicted_audio, target_audio
        # Mel mode:     loss_mel, None, None, mel_pred, mel_target
        loss_total, loss_sc, loss_mag, _, _ = self._compute_loss_and_target(batch)

        # Handle potential error case where loss is high (e.g., shape mismatch in mel loss)
        # The dummy loss from audio loss failure is already handled inside _compute_loss_and_target
        if loss_total is None: # Should not happen if _compute_loss_and_target returns dummy loss
             logger.error("Training step received None loss. Skipping.")
             return None
        elif isinstance(loss_total, torch.Tensor) and loss_total.item() > 1e8:
             logger.warning("Skipping training step due to high loss value (potential error).")
             # Return the high dummy loss, Pytorch Lightning might handle it gracefully or log it.
             # Don't return None, as Lightning expects a loss tensor or dict.
             return loss_total

        # Determine batch size safely
        batch_size = batch.get('f0', batch.get('mel_spec', None))
        batch_size = batch_size.size(0) if batch_size is not None else 1 # Fallback to 1 if keys missing

        # Log the main loss (either total audio loss or mel loss)
        self.log('train_loss', loss_total, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # Log audio loss components if applicable
        if self.current_stage == 1 and self.use_stage1_vocoder and loss_sc is not None and loss_mag is not None:
            self.log('train_loss_sc', loss_sc, prog_bar=False, sync_dist=True, batch_size=batch_size)
            self.log('train_loss_mag', loss_mag, prog_bar=False, sync_dist=True, batch_size=batch_size)

        return loss_total

    def validation_step(self, batch, batch_idx):
        # Unpack potentially different return values based on mode
        loss_total, loss_sc, loss_mag, prediction, target = self._compute_loss_and_target(batch)

        # Handle potential error case
        if loss_total is None:
             logger.error("Validation step received None loss. Skipping.")
             return None
        elif isinstance(loss_total, torch.Tensor) and loss_total.item() > 1e8:
             logger.warning("Skipping validation step due to high loss value (potential error).")
             self.log('val_loss', loss_total, prog_bar=True, sync_dist=True, batch_size=batch.get('f0', batch.get('mel_spec', None)).size(0))
             return loss_total

        # Determine batch size safely
        batch_size = batch.get('f0', batch.get('mel_spec', None))
        batch_size = batch_size.size(0) if batch_size is not None else 1

        # Log the main validation loss
        self.log('val_loss', loss_total, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # --- Logging Section ---
        log_interval = self.config['train'].get('log_spectrogram_every_n_val_epochs', 5)
        should_log = (batch_idx == 0 and self.global_rank == 0 and self.current_epoch % log_interval == 0)
        logger.debug(f"Validation Logging Check: should_log={should_log} (batch_idx={batch_idx}, rank={self.global_rank}, epoch={self.current_epoch}, interval={log_interval}), stage={self.current_stage}, use_vocoder={self.use_stage1_vocoder}")

        if should_log:
            # --- Log Audio (if Stage 1 Vocoder) ---
            if self.current_stage == 1 and self.use_stage1_vocoder:
                if loss_sc is not None: self.log('val_loss_sc', loss_sc, prog_bar=False, sync_dist=True, batch_size=batch_size)
                if loss_mag is not None: self.log('val_loss_mag', loss_mag, prog_bar=False, sync_dist=True, batch_size=batch_size)

                if isinstance(prediction, torch.Tensor) and isinstance(target, torch.Tensor):
                    try:
                        # --- Start New Comparison Logging ---
                        audio_lengths = batch.get('audio_length')
                        length = audio_lengths[0].item() if audio_lengths is not None and len(audio_lengths) > 0 else prediction.shape[1]

                        # 1. Get target mel (assuming batch size >= 1)
                        target_mel_orig = batch['mel_spec'][0] # Shape: [T_orig, F]

                        # 2. Get audio waveforms (already prepared)
                        pred_audio_plot = prediction[0, :length].detach().cpu() # Reuse existing
                        target_audio_plot = target[0, :length].detach().cpu() # Reuse existing (already resampled/trimmed)

                        # 3. Instantiate Mel Transform
                        fmax_val = self.config['audio'].get('fmax')
                        if isinstance(fmax_val, str) and fmax_val.lower() == 'null':
                            fmax_val = None

                        mel_transform = T.MelSpectrogram(
                            sample_rate=self.config['audio']['sample_rate'],
                            n_fft=self.config['audio']['n_fft'],
                            hop_length=self.config['audio']['hop_length'],
                            win_length=self.config['audio']['win_length'],
                            n_mels=self.config['audio']['n_mels'],
                            f_min=self.config['audio']['fmin'],
                            f_max=fmax_val,
                            power=2.0 # Assuming power spectrogram
                        ).to(prediction.device) # Use prediction's device

                        # 4. Compute predicted mel from predicted audio waveform
                        pred_mel = mel_transform(prediction[0, :length]) # Compute on device, Shape: [F, T_mel]

                        # 5. Prepare for plotting
                        pred_mel_plot = pred_mel.cpu().detach() # [F, T_mel]
                        target_mel_plot = target_mel_orig.T.cpu().detach() # [F, T_orig]

                        # Ensure time dimensions match for comparison plot (trim longer to shorter)
                        min_mel_time = min(target_mel_plot.shape[1], pred_mel_plot.shape[1])
                        target_mel_plot = target_mel_plot[:, :min_mel_time]
                        pred_mel_plot = pred_mel_plot[:, :min_mel_time]

                        # 6. Calculate Auto Color Range from Target Mel
                        vmin_auto = target_mel_plot.min().item()
                        vmax_auto = target_mel_plot.max().item()
                        # Add a small buffer if min/max are too close, or handle potential NaN/Inf
                        if not np.isfinite(vmin_auto) or not np.isfinite(vmax_auto) or vmax_auto <= vmin_auto + 1e-6:
                             vmin_auto, vmax_auto = -80, 4 # Fallback to defaults
                        else:
                             # Optional: Add a small margin
                             margin = (vmax_auto - vmin_auto) * 0.05
                             vmin_auto -= margin
                             vmax_auto += margin

                        # 7. Generate and Log Mel Comparison Plot
                        try:
                            fig_mel = plot_spectrograms_comparison_to_figure(
                                target_mel_plot, # [F, T_mel_min]
                                pred_mel_plot,   # [F, T_mel_min]
                                title1="Target Mel",
                                title2="Predicted Mel (from Audio)",
                                main_title=f"Stage 1 Mel Comparison (Step: {self.global_step})",
                                vmin=vmin_auto, # Use auto range
                                vmax=vmax_auto  # Use auto range
                            )
                            self.logger.experiment.add_figure(
                                "Stage_1/val_mel_comparison", # Updated tag
                                fig_mel,
                                self.global_step
                            )
                            plt.close(fig_mel) # Close figure to free memory
                            
                        except Exception as e_mel_comp_log:
                                logger.error(f"Error logging mel comparison plot in validation: {e_mel_comp_log}", exc_info=True)
                                
                        # 8. Log Playable Audio
                        try:
                            if self.stage1_target_sr is not None and self.stage1_target_sr > 0:
                                # Ensure tensors are suitable for add_audio (N, L) and on CPU
                                # Use the 'target' and 'prediction' tensors directly as they are the
                                # resampled/trimmed versions used for loss calculation.
                                target_audio_log = target[0].unsqueeze(0).cpu()
                                pred_audio_log = prediction[0].unsqueeze(0).cpu()

                                self.logger.experiment.add_audio(
                                    "Stage_1/val_audio_target",
                                    target_audio_log,
                                    self.global_step,
                                    sample_rate=int(self.stage1_target_sr) # Ensure sample rate is int
                                )
                                self.logger.experiment.add_audio(
                                    "Stage_1/val_audio_predicted",
                                    pred_audio_log,
                                    self.global_step,
                                    sample_rate=int(self.stage1_target_sr) # Ensure sample rate is int
                                )
                            else:
                                logger.warning("Stage 1 target sample rate not available, skipping audio logging.")
                        except Exception as e_audio_log:
                            logger.error(f"Error logging audio waveforms in validation: {e_audio_log}", exc_info=True)

                        # 9. Generate and Log Waveform Comparison Plot
                        try:
                            if self.stage1_target_sr is not None and self.stage1_target_sr > 0:
                                # Reuse the tensors prepared for add_audio (target_audio_log, pred_audio_log)
                                # Ensure they are 1D for plotting and on CPU
                                target_wav_plot = target_audio_log.squeeze().cpu() # Squeeze potential channel dim & ensure CPU
                                pred_wav_plot = pred_audio_log.squeeze().cpu()   # Squeeze potential channel dim & ensure CPU

                                if target_wav_plot.dim() == 1 and pred_wav_plot.dim() == 1:
                                    fig_wav = plot_waveforms_to_figure(
                                        ground_truth=target_wav_plot,
                                        prediction=pred_wav_plot,
                                        sample_rate=int(self.stage1_target_sr),
                                        title=f"Stage 1 Waveform Comparison (Step: {self.global_step})"
                                    )
                                    self.logger.experiment.add_figure(
                                        "Stage_1/val_waveform_comparison",
                                        fig_wav,
                                        self.global_step
                                    )
                                    plt.close(fig_wav) # Close figure to free memory
                                else:
                                    logger.warning(f"Skipping waveform plot: Waveform dimensions not 1D after squeeze. Target: {target_audio_log.shape}, Pred: {pred_audio_log.shape}")
                            else:
                                logger.warning("Stage 1 target sample rate not available, skipping waveform plot logging.")
                        except Exception as e_wav_plot_log:
                            logger.error(f"Error logging waveform comparison plot in validation: {e_wav_plot_log}", exc_info=True)

                    except Exception as e:
                        logger.error(f"Error during validation comparison logging: {e}", exc_info=True)
                else:
                    logger.warning("Skipping comparison logging due to invalid prediction or target tensors.")

            # --- Log Mel Downsample Comparison (if NOT Stage 1 Vocoder) ---
            elif not (self.current_stage == 1 and self.use_stage1_vocoder):
                if isinstance(prediction, torch.Tensor) and isinstance(target, torch.Tensor) and 'mel_spec' in batch:
                    original_lengths = batch.get('original_length')
                    padded_lengths = batch.get('length')
                    original_mel = batch.get('mel_spec')

                    if original_lengths is not None and len(original_lengths) > 0 and \
                       padded_lengths is not None and len(padded_lengths) > 0 and \
                       original_mel is not None:

                        actual_original_len = original_lengths[0].item()
                        padded_len = padded_lengths[0].item()
                        # Note: 'target' here IS the downsampled mel from _compute_loss_and_target
                        # Its time dimension should already be padded_len // self.downsample_stride
                        downsampled_length = target.shape[1] # Use actual shape of downsampled target

                        # Prepare original mel - Use full padded length
                        original_to_plot = original_mel[0, :padded_len, :].detach().cpu().T # Use padded_len for slicing

                        # Prepare downsampled target mel - SLICE using its own length
                        target_to_plot = target[0, :downsampled_length, :].detach().cpu().T

                        # Pass padded length for title
                        self._log_mel_downsample_comparison(original_to_plot, target_to_plot, padded_len)
                    else:
                        logger.warning("Skipping mel downsample comparison due to missing lengths or original mel.")
                else:
                    logger.warning("Skipping mel downsample comparison due to invalid prediction, target, or missing original mel.")

        return loss_total

    def _log_mel_comparison(self, pred_mel, target_mel):
        """Logs a comparison of predicted and target mel spectrograms to TensorBoard."""
        # This function is likely unused now, but kept for potential future use
        if not isinstance(pred_mel, torch.Tensor) or not isinstance(target_mel, torch.Tensor):
             logger.warning("Cannot log mel comparison: Invalid input types.")
             return
        if pred_mel.shape != target_mel.shape:
             logger.warning(f"Cannot log mel comparison: Shape mismatch. Pred: {pred_mel.shape}, Target: {target_mel.shape}")
             return
        try:
            fig = plot_spectrograms_to_figure(
                ground_truth=target_mel,
                prediction=pred_mel
            )
            self.logger.experiment.add_figure(
                f"Stage_{self.current_stage}/Mel_Comparison_Legacy", # Renamed tag
                fig,
                global_step=self.global_step
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error logging legacy mel comparison: {e}", exc_info=True)

    # Modified to accept padded length for title
    def _log_mel_downsample_comparison(self, original_mel, downsampled_mel, padded_length: int):
        """Logs a comparison of original (padded length) and downsampled target mel spectrograms."""
        if not isinstance(original_mel, torch.Tensor) or not isinstance(downsampled_mel, torch.Tensor):
             logger.warning("Cannot log mel downsample comparison: Invalid input types.")
             return

        try:
            # original_mel is already prepared with padded length in validation_step
            # Input shapes are assumed (F, T_padded) and (F, T_down)

            # Use the new comparison plotting function
            fig = plot_spectrograms_comparison_to_figure(
                spec1=original_mel,    # Use the full padded original mel
                spec2=downsampled_mel,
                title1=f"Original Mel (Padded Length: {padded_length})", # Use padded length in title
                title2=f"Downsampled Target Mel (Length: {downsampled_mel.shape[1]})",
                main_title=f"Stage_{self.current_stage}/Mel_Downsample_Comparison"
            )

            # Log the figure to TensorBoard
            self.logger.experiment.add_figure(
                f"Stage_{self.current_stage}/Mel_Downsample_Comparison",
                fig,
                global_step=self.global_step
            )
            plt.close(fig) # Close the figure to free memory
        except Exception as e:
            logger.error(f"Error logging mel downsample comparison: {e}", exc_info=True)

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        # Select learning rate based on the current stage
        stage_key = f'stage{self.current_stage}'
        try:
            lr = self.config['train']['learning_rate_per_stage'][stage_key]
            logger.info(f"Using learning rate {lr} for stage {self.current_stage}")
        except KeyError:
            default_lr = 0.001 # Define a sensible default
            logger.warning(f"Learning rate for '{stage_key}' not found in config. Using default: {default_lr}")
            lr = default_lr

        # Adam optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=self.config['train'].get('weight_decay', 0.0001)
        )

        # Learning rate scheduler (ReduceLROnPlateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', # Monitor validation loss
            factor=self.config['train'].get('lr_factor', 0.5),
            patience=self.config['train'].get('lr_patience', 10),
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss', # Metric to monitor
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    # --- Checkpoint Loading/Freezing Logic (Moved from train.py) ---
    # Pytorch Lightning handles checkpoint loading via trainer.fit(ckpt_path=...)
    # We might need hooks if we want custom freezing logic *after* loading.

    def on_load_checkpoint(self, checkpoint):
        """Called by Lightning after loading a checkpoint but before fitting."""
        loaded_stage = checkpoint.get('hyper_parameters', {}).get('config', {}).get('model', {}).get('current_stage', None)
        freeze_weights = self.config['train'].get('freeze_loaded_weights', False) # Get freeze setting from *current* config

        logger.info(f"on_load_checkpoint: Loaded checkpoint from stage {loaded_stage}. Current stage: {self.current_stage}. Freeze setting: {freeze_weights}")

        if loaded_stage is not None and loaded_stage < self.current_stage and freeze_weights:
            logger.info(f"Freezing weights from previous stage ({loaded_stage}) as current stage is {self.current_stage}.")
            if loaded_stage >= 1:
                logger.info("Freezing LowResModel.")
                for param in self.low_res_model.parameters():
                    param.requires_grad = False
            if loaded_stage >= 2:
                 logger.info("Freezing MidResUpsampler.")
                 for param in self.mid_res_upsampler.parameters():
                     param.requires_grad = False
            # Stage 3 weights are never frozen when loading for stage 3 training
        else:
             logger.info("Not freezing weights (same stage, next stage, or freeze_loaded_weights=False).")

    # Optional: If you need to ensure weights are trainable when starting a stage *without* loading a checkpoint
    # def on_train_start(self):
    #     """Called before training begins (after setup and checkpoint loading)."""
    #     # Ensure weights for the current stage and subsequent stages are trainable
    #     logger.info(f"on_train_start: Ensuring weights for stage {self.current_stage} and beyond are trainable.")
    #     if self.current_stage <= 1:
    #         for param in self.low_res_model.parameters(): param.requires_grad = True
    #     if self.current_stage <= 2:
    #         for param in self.mid_res_upsampler.parameters(): param.requires_grad = True
    #     # Stage 3 (high_res_upsampler) should always be trainable when reached
    #     for param in self.high_res_upsampler.parameters(): param.requires_grad = True
