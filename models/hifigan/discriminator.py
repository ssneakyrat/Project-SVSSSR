import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

# Placeholder for LRELU_SLOPE, will get from config later
LRELU_SLOPE = 0.1

class DiscriminatorP(nn.Module):
    """ Multi-Period Discriminator Period Module """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(int((kernel_size - 1)/2), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(int((kernel_size - 1)/2), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(int((kernel_size - 1)/2), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(int((kernel_size - 1)/2), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(int((kernel_size - 1)/2), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        # x: [B, 1, T]
        # Reshape for 2D convolution
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period) # [B, 1, T/period, period]

        feat = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1) # Flatten to [B, N]

        return x, feat # Return final score and intermediate features

class MultiPeriodDiscriminator(nn.Module):
    """ Multi-Period Discriminator """
    def __init__(self, config):
        super().__init__()
        h = config['vocoder']['hifigan']
        periods = h.get('discriminator_periods', [2, 3, 5, 7, 11])
        use_spectral_norm = h.get('discriminator_use_spectral_norm', False)
        
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(self, y, y_hat):
        # y: real audio [B, 1, T]
        # y_hat: generated audio [B, 1, T]
        y_d_rs = [] # Real scores
        y_d_gs = [] # Generated scores
        fmap_rs = [] # Real feature maps
        fmap_gs = [] # Generated feature maps
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    """ Multi-Scale Discriminator Scale Module """
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        # x: [B, 1, T]
        feat = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1) # Flatten to [B, N]

        return x, feat # Return final score and intermediate features

class MultiScaleDiscriminator(nn.Module):
    """ Multi-Scale Discriminator """
    def __init__(self, config):
        super().__init__()
        h = config['vocoder']['hifigan']
        use_spectral_norm = h.get('discriminator_use_spectral_norm', False)
        
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
            DiscriminatorS(), # Downsample by 2
            DiscriminatorS(), # Downsample by 4
        ])
        # Pooling layers for downsampling input audio
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        # y: real audio [B, 1, T]
        # y_hat: generated audio [B, 1, T]
        y_d_rs = [] # Real scores
        y_d_gs = [] # Generated scores
        fmap_rs = [] # Real feature maps
        fmap_gs = [] # Generated feature maps

        # Process original scale
        y_d_r, fmap_r = self.discriminators[0](y)
        y_d_g, fmap_g = self.discriminators[0](y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs.append(fmap_r)
        y_d_gs.append(y_d_g)
        fmap_gs.append(fmap_g)

        # Process downsampled scales
        for i, d in enumerate(self.discriminators[1:]):
            y = self.meanpools[i](y)
            y_hat = self.meanpools[i](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs