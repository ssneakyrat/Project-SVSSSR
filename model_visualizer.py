import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class ModelVisualizer:
    """
    Visualization and logging hooks for the Attention-Free SVS model
    to help debug and analyze the model's behavior.
    """
    def __init__(self, model, log_dir='logs/visualizations'):
        """
        Initialize the visualizer with a model instance and logging directory.
        
        Args:
            model: An instance of AttentionFreeSVS
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        self.hooks = []
        self.intermediate_outputs = {}
        
        # Register hooks for all components
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for all model components we want to visualize"""
        # 1. Input Processing - Phone Encoder
        self._register_hook(self.model.phone_encoder, 'phone_encoder')
        self._register_hook(self.model.f0_encoder, 'f0_encoder')
        self._register_hook(self.model.duration_encoder, 'duration_encoder')
        self._register_hook(self.model.midi_encoder, 'midi_encoder')
        
        # 2. Dilated Backbone
        self._register_hook(self.model.backbone, 'backbone')
        
        # 3. Feature Mixer
        self._register_hook(self.model.feature_mixer, 'feature_mixer')
        
        # 4. Decoder
        self._register_hook(self.model.decoder, 'decoder')
        
        # Register hooks for internal decoder components (upsampling)
        for i, layer in enumerate(self.model.decoder.upsample_layers):
            self._register_hook(layer, f'decoder_upsample_{i}')
        
        # Register hook for final projection
        self._register_hook(self.model.decoder.final_proj, 'mel_projection')
    
    def _register_hook(self, module, name):
        """Register a forward hook for a specific module"""
        def hook_fn(module, input, output):
            self.intermediate_outputs[name] = output
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def log_inputs(self, phone_ids, f0, durations, midi_ids, step=None):
        """Log model inputs to TensorBoard"""
        if step is not None:
            self.step = step
        
        # Log phone IDs as a heatmap
        self._log_tensor_heatmap(phone_ids, 'inputs/phone_ids')
        
        # Log F0 contour as a line plot
        self._log_f0_contour(f0.squeeze(-1), 'inputs/f0_contour')
        
        # Log durations as a bar chart
        self._log_durations(durations.squeeze(-1), 'inputs/durations')
        
        # Log MIDI IDs as piano roll
        self._log_tensor_heatmap(midi_ids, 'inputs/midi_ids')
    
    def log_outputs(self, mel_pred, mel_target=None, step=None):
        """Log model outputs and intermediate representations to TensorBoard"""
        if step is not None:
            self.step = step
        
        # Log final mel prediction
        self._log_spectrogram(mel_pred, 'outputs/mel_prediction')
        
        if mel_target is not None:
            # Log target mel
            self._log_spectrogram(mel_target, 'outputs/mel_target')
            
            # Log difference between prediction and target
            self._log_mel_diff(mel_pred, mel_target, 'outputs/mel_difference')
        
        # Log intermediate outputs from each stage
        self._log_intermediate_outputs()
        
        # Increment step counter
        self.step += 1
    
    def _log_intermediate_outputs(self):
        """Log all intermediate outputs collected from hooks"""
        # 1. Input Processing Stage outputs
        for name in ['phone_encoder', 'f0_encoder', 'duration_encoder', 'midi_encoder']:
            if name in self.intermediate_outputs:
                features = self.intermediate_outputs[name]
                # Log activations as heatmap
                self._log_feature_activations(features, f'stage1_input_processing/{name}')
        
        # 2. Backbone output
        if 'backbone' in self.intermediate_outputs:
            backbone_out = self.intermediate_outputs['backbone']
            self._log_feature_activations(backbone_out, 'stage2_backbone/output')
        
        # 3. Feature mixer output
        if 'feature_mixer' in self.intermediate_outputs:
            mixer_out = self.intermediate_outputs['feature_mixer']
            self._log_feature_activations(mixer_out, 'stage3_feature_mixer/output')
        
        # 4. Decoder internal stages and final output
        # Upsampling layers
        for i in range(len(self.model.decoder.upsample_layers)):
            name = f'decoder_upsample_{i}'
            if name in self.intermediate_outputs:
                upsample_out = self.intermediate_outputs[name]
                # Convert from [batch, channels, time] to appropriate format for visualization
                if isinstance(upsample_out, torch.Tensor) and upsample_out.dim() >= 3:
                    # Take first item in batch
                    upsample_sample = upsample_out[0] if upsample_out.dim() > 2 else upsample_out
                    # Log as heatmap
                    self._log_tensor_heatmap(upsample_sample, f'stage4_decoder/{name}')
        
        # Final projection (before tanh)
        if 'mel_projection' in self.intermediate_outputs:
            proj_out = self.intermediate_outputs['mel_projection']
            if isinstance(proj_out, torch.Tensor) and proj_out.dim() >= 3:
                # Take first item in batch
                proj_sample = proj_out[0] if proj_out.dim() > 2 else proj_out
                # Log as spectrogram
                self._log_tensor_heatmap(proj_sample, 'stage4_decoder/pre_tanh_projection')
                
                # Also log the histograms of values before tanh
                self.writer.add_histogram('stage4_decoder/pre_tanh_values', proj_out, self.step)
    
    def _log_feature_activations(self, features, tag):
        """Log feature activations as heatmaps and histograms"""
        if isinstance(features, torch.Tensor):
            # Use the first item in the batch
            if features.dim() > 2:
                features = features[0]
            
            # If features is [time, channels], transpose for better visualization
            if features.dim() == 2 and features.shape[0] > features.shape[1]:
                features = features.transpose(0, 1)
            
            # Log as heatmap
            self._log_tensor_heatmap(features, f'{tag}/heatmap')
            
            # Log histogram of activation values
            self.writer.add_histogram(f'{tag}/activation_values', features, self.step)
            
            # Log mean activations per channel/feature
            if features.dim() >= 2:
                mean_acts = torch.mean(features, dim=-1)  # Average over time
                fig = plt.figure(figsize=(10, 4))
                plt.bar(range(len(mean_acts)), mean_acts.cpu().detach().numpy())
                plt.title('Mean Activation Per Feature')
                plt.xlabel('Feature Index')
                plt.ylabel('Mean Activation')
                self.writer.add_figure(f'{tag}/mean_activations', fig, self.step)
                plt.close(fig)
    
    def _log_tensor_heatmap(self, tensor, tag):
        """Log a tensor as a heatmap image"""
        if not isinstance(tensor, torch.Tensor):
            return
        
        # Detach and move to CPU
        tensor = tensor.detach().cpu()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(tensor, aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        plt.title(tag.split('/')[-1])
        
        # Add to TensorBoard
        self.writer.add_figure(tag, fig, self.step)
        plt.close(fig)
    
    def _log_spectrogram(self, mel, tag):
        """Log a mel spectrogram"""
        if not isinstance(mel, torch.Tensor):
            return
            
        # Handle different tensor shapes
        if mel.dim() == 4:  # [batch, channels, freq, time]
            mel = mel[0, 0]  # Take first batch, first channel
        elif mel.dim() == 3:  # [batch, freq, time] or [channels, freq, time]
            mel = mel[0]  # Take first item
            
        # Detach and move to CPU
        mel = mel.detach().cpu().numpy()
        
        # Create spectrogram plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mel, aspect='auto', origin='lower', interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.title(f'{tag.split("/")[-1]}')
        plt.tight_layout()
        
        # Add to TensorBoard
        self.writer.add_figure(tag, fig, self.step)
        plt.close(fig)
    
    def _log_mel_diff(self, mel_pred, mel_target, tag):
        """Log difference between predicted and target mel spectrograms"""
        if not isinstance(mel_pred, torch.Tensor) or not isinstance(mel_target, torch.Tensor):
            return
            
        # Handle different tensor shapes and ensure they match
        if mel_pred.dim() == 4:  # [batch, channels, freq, time]
            mel_pred = mel_pred[0, 0]  # Take first batch, first channel
        elif mel_pred.dim() == 3:  # [batch, freq, time] or [channels, freq, time]
            mel_pred = mel_pred[0]  # Take first item
            
        if mel_target.dim() == 4:
            mel_target = mel_target[0, 0]
        elif mel_target.dim() == 3:
            mel_target = mel_target[0]
            
        # Detach and move to CPU
        mel_pred = mel_pred.detach().cpu()
        mel_target = mel_target.detach().cpu()
        
        # Match dimensions if needed (e.g., due to upsampling)
        if mel_pred.shape != mel_target.shape:
            # Use the model's adjustment method
            mel_pred = self.model.adjust_mel_length(mel_pred, mel_target.shape[-1])
            if mel_pred.dim() > mel_target.dim():
                mel_pred = mel_pred.squeeze()
        
        # Calculate absolute difference
        diff = torch.abs(mel_pred - mel_target)
        
        # Create difference plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(diff.numpy(), aspect='auto', origin='lower', cmap='hot', interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.title('Absolute Difference: Prediction vs Target')
        plt.tight_layout()
        
        # Add to TensorBoard
        self.writer.add_figure(tag, fig, self.step)
        plt.close(fig)
        
        # Also log MSE per frequency bin
        mse_per_bin = torch.mean(diff ** 2, dim=1)  # Average over time
        fig = plt.figure(figsize=(10, 4))
        plt.bar(range(len(mse_per_bin)), mse_per_bin.numpy())
        plt.title('MSE Per Frequency Bin')
        plt.xlabel('Mel Bin')
        plt.ylabel('Mean Squared Error')
        self.writer.add_figure(f'{tag}/mse_per_bin', fig, self.step)
        plt.close(fig)
    
    def _log_f0_contour(self, f0, tag):
        """Log F0 contour as a line plot"""
        if not isinstance(f0, torch.Tensor):
            return
            
        # Use the first item in the batch
        if f0.dim() > 1:
            f0 = f0[0]
            
        # Detach and move to CPU
        f0 = f0.detach().cpu().numpy()
        
        # Create F0 contour plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f0)
        plt.title('F0 Contour')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency (Hz)')
        
        # Add to TensorBoard
        self.writer.add_figure(tag, fig, self.step)
        plt.close(fig)
    
    def _log_durations(self, durations, tag):
        """Log durations as a bar chart"""
        if not isinstance(durations, torch.Tensor):
            return
            
        # Use the first item in the batch
        if durations.dim() > 1:
            durations = durations[0]
            
        # Detach and move to CPU
        durations = durations.detach().cpu().numpy()
        
        # Create duration bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(durations)), durations)
        plt.title('Phone Durations')
        plt.xlabel('Phone Index')
        plt.ylabel('Duration (frames)')
        
        # Add to TensorBoard
        self.writer.add_figure(tag, fig, self.step)
        plt.close(fig)


class VisualizedAttentionFreeSVS(nn.Module):
    """
    Wrapper for AttentionFreeSVS that applies visualization during forward pass
    """
    def __init__(self, model, log_dir='logs/visualizations'):
        super().__init__()
        self.model = model
        self.visualizer = ModelVisualizer(model, log_dir)
    
    def forward(self, phone_ids, f0, durations, midi_ids, mel_target=None):
        # Log inputs
        self.visualizer.log_inputs(phone_ids, f0, durations, midi_ids)
        
        # Forward pass through the model
        mel_pred = self.model(phone_ids, f0, durations, midi_ids)
        
        # Log outputs
        self.visualizer.log_outputs(mel_pred, mel_target)
        
        return mel_pred
    
    def __del__(self):
        # Clean up hooks when the object is deleted
        self.visualizer.remove_hooks()


def apply_to_model(model, log_dir='logs/visualizations'):
    """
    Apply visualization to an existing model
    
    Args:
        model: An instance of AttentionFreeSVS
        log_dir: Directory for TensorBoard logs
        
    Returns:
        VisualizedAttentionFreeSVS: The wrapped model
    """
    return VisualizedAttentionFreeSVS(model, log_dir)