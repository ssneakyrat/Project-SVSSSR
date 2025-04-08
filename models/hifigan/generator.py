import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np

# Placeholder for LRELU_SLOPE, will get from config later
LRELU_SLOPE = 0.1

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                               padding=int((kernel_size*d - d)/2)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=int((kernel_size - 1)/2)))
            for _ in range(len(dilation)) # One conv2 for each conv1
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x # Residual connection
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config['vocoder']['hifigan'] # Get HiFiGAN specific config
        self.num_kernels = len(h['resblock_kernel_sizes'])
        self.num_upsamples = len(h['upsample_rates'])
        
        # Initial convolution layer
        # Input channels: mel_bins + f0_embed_dim if use_f0 else mel_bins
        self.mel_bins = config['model']['mel_bins']
        self.use_f0 = config['vocoder'].get('use_f0_conditioning', False)
        input_channels = self.mel_bins
        if self.use_f0:
             # Assuming F0 is embedded upstream or needs embedding here
             # Let's assume it comes in as 1 channel and we embed it
             self.f0_embed_dim = h.get('f0_embed_dim', 64) # Add f0_embed_dim to config
             self.f0_embedding = nn.Conv1d(1, self.f0_embed_dim, kernel_size=1)
             input_channels += self.f0_embed_dim
        
        self.conv_pre = weight_norm(nn.Conv1d(input_channels, h['upsample_initial_channel'], 7, 1, padding=3))

        # Upsampling layers with residual blocks
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h['upsample_rates'], h['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(h['upsample_initial_channel']//(2**i), h['upsample_initial_channel']//(2**(i+1)),
                                   k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h['upsample_initial_channel']//(2**(i+1))
            for j, (k, d) in enumerate(zip(h['resblock_kernel_sizes'], h['resblock_dilation_sizes'])):
                self.resblocks.append(ResBlock(ch, k, d))

        # Final convolution layer
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(self.init_weights)
        self.conv_post.apply(self.init_weights)
        if self.use_f0:
            self.f0_embedding.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, f0=None):
        # x: mel spectrogram [B, M, T_mel]
        # f0: fundamental frequency [B, 1, T_mel] (optional)

        if self.use_f0:
            if f0 is None:
                raise ValueError("F0 conditioning enabled but f0 not provided to Generator")
            f0_emb = self.f0_embedding(f0)
            x = torch.cat([x, f0_emb], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels # Average output of resblocks
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        # Output: [B, 1, T_audio]
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)