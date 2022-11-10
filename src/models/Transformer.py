import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import os
import sys
import math
import ipdb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{ROOT_DIR}/utils")
sys.path.append(CURR_DIR)

from FeedForward import FeedForward
from PositionalEncoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        ntoken: int,
        input_size: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers_encoder: int,
        nlayers_decoder: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        assert nlayers_decoder >= 2
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers_encoder)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = FeedForward(
            d_model * input_size, output_size, nlayers_decoder, d_hid
        )

        self.init_weights()
        self.mask = self.generate_square_subsequent_mask(input_size)

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src_mask = self.mask if src_mask is None else src_mask
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = torch.reshape(
            output, (output.shape[0], output.shape[1] * output.shape[2])
        )
        output = self.decoder(output)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
        return self
