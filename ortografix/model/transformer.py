"""Transformer."""
import math
import logging

import torch

__all__ = ('TEncoder', 'TDecoder')

logger = logging.getLogger(__name__)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# pylint: disable=R0902
class TDecoder(torch.nn.Module):
    """Transformer Decoder class."""

    def __init__(self, hidden_size, output_size, num_layers, dropout,
                 num_attention_heads):
        """Initialize decoder model."""
        super(TDecoder, self).__init__()
        self.model_type = 'transformer'
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
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
        # self.out.bias.data.zero_()
        # self.out.weight.data.uniform_(-0.1, 0.1)

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

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 num_attention_heads):
        """Initialize encoder model."""
        super(TEncoder, self).__init__()
        self.model_type = 'transformer'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_seq_len=15)
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads,
            dim_feedforward=hidden_size, dropout=dropout,
            activation='relu')
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        # TODO: check if we need this
        # self.embedding.weight.data.uniform_(-0.1, 0.1)

    # pylint: disable=R1710, W0221
    def forward(self, input_tensor):
        """Apply forward computation."""
        embedded_tensor = self.embedding(input_tensor).view(1, 1, -1)
        # embedded_tensor = embedded_tensor * math.sqrt(self.hidden_size)
        # embedded_tensor = self.pos_encoder(embedded_tensor)
        return self.transformer(embedded_tensor)
