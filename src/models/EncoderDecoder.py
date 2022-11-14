import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self
