import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvPath(nn.Module):
    """
    Path with multiple dilated convolutions with different dilation rates
    """
    def __init__(self, input_dim, output_dim, kernel_sizes, dilation_rates):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.output_dim = output_dim
        
        # Initial projection to match output_dim
        self.input_proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        
        # Create dilated conv layers
        for i, (ks, dil) in enumerate(zip(kernel_sizes, dilation_rates)):
            # Calculate padding to maintain sequence length
            padding = (ks - 1) * dil // 2
            
            self.layers.append(nn.Conv1d(
                output_dim, 
                output_dim, 
                kernel_size=ks, 
                dilation=dil,
                padding=padding
            ))
            
            self.norms.append(nn.InstanceNorm1d(output_dim))
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, time, channels]
        Returns:
            Path output [batch, time, output_dim]
        """
        # [batch, time, channels] -> [batch, channels, time]
        x = x.transpose(1, 2)
        
        # Project input to output_dim
        x = self.input_proj(x)
        
        # Residual path
        residual = x
        
        # Apply dilated convolution layers
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            # Apply convolution
            conv_out = conv(residual)
            conv_out = norm(conv_out)
            conv_out = F.relu(conv_out)
            
            # Add residual connection
            residual = residual + conv_out
        
        # [batch, output_dim, time] -> [batch, time, output_dim]
        output = residual.transpose(1, 2)
        
        return output


class MultiPathDilatedBackbone(nn.Module):
    """
    Multi-path dilated convolution backbone with varying receptive fields
    """
    def __init__(self, input_dim, num_paths=3, 
                 path1_config=None, path2_config=None, path3_config=None):
        super().__init__()
        
        # Create paths with different dilation configurations
        self.paths = nn.ModuleList()
        
        # Path 1 - Small dilation rates for local patterns
        if path1_config:
            self.paths.append(DilatedConvPath(
                input_dim=input_dim,
                output_dim=path1_config['channels'],
                kernel_sizes=path1_config['kernel_sizes'],
                dilation_rates=path1_config['dilation_rates']
            ))
        
        # Path 2 - Medium dilation rates for mid-range patterns
        if path2_config and num_paths >= 2:
            self.paths.append(DilatedConvPath(
                input_dim=input_dim,
                output_dim=path2_config['channels'],
                kernel_sizes=path2_config['kernel_sizes'],
                dilation_rates=path2_config['dilation_rates']
            ))
        
        # Path 3 - Large dilation rates for long-range patterns
        if path3_config and num_paths >= 3:
            self.paths.append(DilatedConvPath(
                input_dim=input_dim,
                output_dim=path3_config['channels'],
                kernel_sizes=path3_config['kernel_sizes'],
                dilation_rates=path3_config['dilation_rates']
            ))
        
        # Calculate total output dimensions
        self.output_dim = sum(path.output_dim for path in self.paths)
        
        # MLP after concatenation
        self.output_mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, time, channels]
        Returns:
            Backbone output [batch, time, output_dim]
        """
        # Process input through each path
        path_outputs = []
        for path in self.paths:
            path_outputs.append(path(x))
        
        # Concatenate outputs along feature dimension
        concatenated = torch.cat(path_outputs, dim=-1)
        
        # Apply output MLP
        output = self.output_mlp(concatenated)
        
        return output