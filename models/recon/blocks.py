import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=se_reduction)
        
    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = self.bn2(self.conv2(conv1))
        
        residual = x if self.skip_proj is None else self.skip_proj(x)
        out = F.relu(conv2 + residual)
        
        if self.use_se:
            out = self.se(out)
            
        skip = out
        pooled = self.pool(out)
        
        return pooled, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, se_reduction=16):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=se_reduction)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        
        residual = x
        x = self.bn2(self.conv2(x))
        
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
            
        x = F.relu(x + residual)
        
        if self.use_se:
            x = self.se(x)
            
        return x

class DualPathBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.bg_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.fg_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x if self.project is None else self.project(x)
        
        bg_features = self.bg_path(x)
        fg_features = self.fg_path(x)
        
        concat_features = torch.cat([bg_features, fg_features], dim=1)
        
        attention_weights = self.attention(concat_features)
        bg_weight = attention_weights[:, 0:1, :, :]
        fg_weight = attention_weights[:, 1:2, :, :]
        
        out = bg_weight * bg_features + fg_weight * fg_features
        
        return F.relu(out + residual)

class EncoderBlockDualPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dual_path = DualPathBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.dual_path(x)
        skip = out
        pooled = self.pool(out)
        return pooled, skip

class DecoderBlockDualPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv_reduce = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.dual_path = DualPathBlock(in_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_reduce(x)
        x = self.dual_path(x)
        
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded'):
        super().__init__()
        
        self.in_channels = in_channels
        self.mode = mode
        
        if inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        else:
            self.inter_channels = inter_channels
            
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        if mode == 'gaussian':
            self.theta = None
            self.phi = None
        elif mode == 'concatenation':
            self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1),
                nn.ReLU()
            )
        else:  # embedded or dot
            self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        nn.init.constant_(self.W[0].weight, 0)
        nn.init.constant_(self.W[0].bias, 0)
        
        self.scale_factor = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x)
        g_x = g_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        
        if self.mode == 'gaussian':
            theta_x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            
        elif self.mode == 'concatenation':
            theta_x = self.theta(x)
            phi_x = self.phi(x)
            
            theta_x = theta_x.view(batch_size, self.inter_channels, -1, 1)
            phi_x = phi_x.view(batch_size, self.inter_channels, 1, -1)
            
            concat_feature = torch.cat([theta_x.repeat(1, 1, 1, phi_x.size(3)), 
                                      phi_x.repeat(1, 1, theta_x.size(2), 1)], dim=1)
                
            f = self.concat_project(concat_feature)
            f = f.view(batch_size, -1, phi_x.size(3))
            f_div_C = F.softmax(f, dim=-1)
            
        else:  # embedded or dot
            theta_x = self.theta(x)
            theta_x = theta_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            
            phi_x = self.phi(x)
            phi_x = phi_x.view(batch_size, self.inter_channels, -1)
            
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        z = self.scale_factor * W_y + x
        
        return z

class DilatedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.dil1 = nn.Conv2d(out_channels, out_channels//4, 3, padding=1, dilation=1)
        self.dil2 = nn.Conv2d(out_channels, out_channels//4, 3, padding=2, dilation=2)
        self.dil3 = nn.Conv2d(out_channels, out_channels//4, 3, padding=4, dilation=4)
        self.dil4 = nn.Conv2d(out_channels, out_channels//4, 3, padding=8, dilation=8)
        
        self.bn_dil = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(in_channels)
        
        self.attention = SelfAttention(in_channels, num_heads=num_heads)
        
    def forward(self, x):
        residual = x
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        d1 = F.relu(self.dil1(x))
        d2 = F.relu(self.dil2(x))
        d3 = F.relu(self.dil3(x))
        d4 = F.relu(self.dil4(x))
        
        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = F.relu(self.bn_dil(out))
        
        out = self.bn_out(self.conv_out(out))
        out = F.relu(out + residual)
        out = self.attention(out)
        
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        query = self.query_proj(x).view(batch_size, self.num_heads, self.head_dim, -1)
        key = self.key_proj(x).view(batch_size, self.num_heads, self.head_dim, -1)
        value = self.value_proj(x).view(batch_size, self.num_heads, self.head_dim, -1)
        
        query = query.permute(0, 1, 3, 2)
        
        attention = torch.matmul(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, value.permute(0, 1, 3, 2))
        
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, H, W)
        out = self.output_proj(out)
        
        return self.gamma * out + x

class LowFrequencyEmphasis(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, mel_bins=80):
        super().__init__()
        self.in_channels = in_channels
        self.mel_bins = mel_bins
        
        self.freq_pool = nn.AvgPool2d(kernel_size=(1, mel_bins), stride=1)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        freq_bias = torch.ones(1, mel_bins)
        for i in range(mel_bins):
            freq_bias[0, i] = 1.0 - (i / mel_bins) * 0.8
        
        self.freq_bias = nn.Parameter(freq_bias)
    
    def forward(self, x):
        batch_size, channels, time_steps, freq_bins = x.size()
        
        if freq_bins != self.mel_bins:
            device = x.device
            new_bias = torch.ones(1, freq_bins, device=device)
            for i in range(freq_bins):
                new_bias[0, i] = 1.0 - (i / freq_bins) * 0.8
            freq_bias = new_bias
        else:
            freq_bias = self.freq_bias
        
        if freq_bins != self.mel_bins:
            pooled = F.avg_pool2d(x, kernel_size=(1, freq_bins))
        else:
            pooled = self.freq_pool(x)
        
        attn = self.channel_attention(pooled)
        attn = attn.expand(-1, -1, -1, freq_bins)
        
        expanded_bias = freq_bias.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, time_steps, -1)
        
        return x * attn * expanded_bias