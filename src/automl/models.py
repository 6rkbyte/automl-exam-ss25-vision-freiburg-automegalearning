import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN_flex(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, params: dict):
        super().__init__()
        self.layers = nn.ModuleList()

        # Get architecture params
        n_layers = params["linear_n_layers"]
        conv_n_layers = params["conv_n_layers"]

        # Initial conv layers
        channels = in_channels
        for _ in range(conv_n_layers):
            self.layers.append(nn.Conv2d(channels, channels * 3, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.GELU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            channels *= 3

        self.layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.AdaptiveMaxPool2d((12, 12)))
        self.layers.append(nn.Flatten())
        channels *= 2
        in_features = 12 * 12 * channels

        # Fully connected layers
        for i in range(n_layers):
            out_features = params[f"n_units_l{i}"]
            #dropout = params[f"dropout_l{i}"]
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            #self.layers.append(nn.Dropout(dropout))
            in_features = out_features

        # Output layer
        self.layers.append(nn.Linear(in_features, num_classes))
        self.layers.append(nn.LogSoftmax(dim=1))

        # Wrap in a Sequential
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)