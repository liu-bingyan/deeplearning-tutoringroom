import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # Padding on the left only
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels,
                 kernel_size, dilation, global_condition_channels=None):
        super().__init__()
        self.filter_conv = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)

        self.gc_filter_proj = None
        self.gc_gate_proj = None
        if global_condition_channels is not None:
            self.gc_filter_proj = nn.Conv1d(global_condition_channels, dilation_channels, 1)
            self.gc_gate_proj = nn.Conv1d(global_condition_channels, dilation_channels, 1)

        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)

    def forward(self, x, gc=None):
        conv_filter = self.filter_conv(x)
        conv_gate = self.gate_conv(x)

        if gc is not None:
            conv_filter += self.gc_filter_proj(gc)
            conv_gate += self.gc_gate_proj(gc)

        z = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        skip = self.skip_conv(z)
        residual = self.residual_conv(z) + x  # Residual connection
        print(residual.shape, skip.shape)  # Debugging output
        return residual, skip

class WaveNet(nn.Module):
    def __init__(self, in_channels, residual_channels, dilation_channels,
                 skip_channels, end_channels, kernel_size, dilations,
                 global_condition_channels=None, quantization_channels=256):
        super().__init__()
        self.quantization_channels = quantization_channels
        self.initial_conv = CausalConv1d(in_channels, residual_channels, kernel_size=1)

        self.dilated_blocks = nn.ModuleList()
        for dilation in dilations:
            self.dilated_blocks.append(
                ResidualBlock(residual_channels, dilation_channels, skip_channels,
                              kernel_size, dilation, global_condition_channels)
            )

        self.relu1 = nn.ReLU()
        self.conv_post_1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.relu2 = nn.ReLU()
        self.conv_post_2 = nn.Conv1d(end_channels, quantization_channels, 1)

    def forward(self, x, gc=None):
        x = self.initial_conv(x)
        skip_connections = []

        for block in self.dilated_blocks:
            x, skip = block(x, gc)
            skip_connections.append(skip)

        out = sum(skip_connections)
        out = self.relu1(out)
        out = self.conv_post_1(out)
        out = self.relu2(out)
        out = self.conv_post_2(out)
        return out


if __name__ == "__main__":
    # Example usage
    batch_size = 16
    in_channels = 1
    residual_channels = 32
    dilation_channels = 32
    skip_channels = 32
    end_channels = 32
    kernel_size = 2
    dilations = [1, 2, 4, 8, 16]
    global_condition_channels = None
    quantization_channels = 256
    seq_length = 16000  # Example sequence length
    x = torch.randn(batch_size, in_channels, seq_length)
    gc = None  # Global conditioning can be None for this example

    model = WaveNet(in_channels, residual_channels, dilation_channels,
                    skip_channels, end_channels, kernel_size, dilations,
                    global_condition_channels, quantization_channels)
    output = model(x, gc)
    print("Output shape:", output.shape)
    # Output shape: torch.Size([16, 256, 16000])