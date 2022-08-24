import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, num_layers: int, layer_dim: int
    ):
        super().__init__()
        assert num_layers >= 2
        self.layers = [nn.Linear(input_size, layer_dim)]
        for _ in range(num_layers):
            self.layers.append(nn.Linear(layer_dim, layer_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layer_dim, output_size))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
