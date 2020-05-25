"""Transformer."""
import logging

import torch

__all__ = ('TEncoder', 'TDecoder')

logger = logging.getLogger(__name__)


# pylint: disable=R0902
class TDecoder(torch.nn.Module):
    """Transformer Decoder class."""

    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0,
                 num_attention_heads=2):
        """Initialize decoder model."""
        super(TDecoder, self).__init__()
        self.model_type = 'transformer'
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        decoder_layers = torch.nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_attention_heads,
            dim_feedforward=hidden_size, dropout=dropout,
            activation='relu')
        self.transformer = torch.nn.TransformerDecoder(
            decoder_layers, num_layers=num_layers)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        # TODO check that we need this
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-0.1, 0.1)

    # pylint: disable=R1710, W0221
    def forward(self, input_tensor, memory_tensor):
        """Apply forward computation."""
        output_tensor = self.embedding(input_tensor).view(1, 1, -1)
        # TODO: maybe remove this for the Transformer?
        output_tensor = torch.nn.functional.relu(output_tensor)
        output_tensor = self.transformer(output_tensor, memory_tensor)
        output_tensor = self.softmax(self.out(output_tensor[0]))
        return output_tensor


# pylint: disable=R0902
class TEncoder(torch.nn.Module):
    """Transformer Encoder class."""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
                 num_attention_heads=2):
        """Initialize encoder model."""
        super(TEncoder, self).__init__()
        self.model_type = 'transformer'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads,
            dim_feedforward=hidden_size, dropout=dropout,
            activation='relu')
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        # TODO: check if we need this
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    # pylint: disable=R1710, W0221
    def forward(self, input_tensor):
        """Apply forward computation."""
        embedded_tensor = self.embedding(input_tensor).view(1, 1, -1)
        return self.transformer(embedded_tensor)
