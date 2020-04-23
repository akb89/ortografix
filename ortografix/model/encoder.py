"""Encoder."""
import logging

import torch

import ortografix.utils.constants as const

__all__ = ('Encoder')

logger = logging.getLogger(__name__)


# pylint: disable=R0902
class Encoder(torch.nn.Module):
    """Encoder class."""

    def __init__(self, model_type, input_size, hidden_size, num_layers=1,
                 nonlinearity='tanh', bias=True, dropout=0,
                 bidirectional=False):
        """Initialize encoder model."""
        if model_type not in ['rnn', 'gru', 'lstm']:
            raise Exception('Unsupported model type: {}'.format(model_type))
        super(Encoder, self).__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        if self.model_type == 'rnn':
            self.rnn = torch.nn.RNN(
                input_size=hidden_size, hidden_size=hidden_size,
                num_layers=num_layers, nonlinearity=nonlinearity,
                bias=bias, batch_first=False, dropout=dropout,
                bidirectional=bidirectional)
        if self.model_type == 'gru':
            self.gru = torch.nn.GRU(
                input_size=hidden_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=bias, batch_first=False,
                dropout=dropout, bidirectional=bidirectional)
        if self.model_type == 'lstm':
            self.lstm = torch.nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=bias, batch_first=False,
                dropout=dropout, bidirectional=bidirectional)

    # pylint: disable=R1710, W0221
    def forward(self, input_tensor, hidden_tensor):
        """Apply forward computation."""
        embedded_tensor = self.embedding(input_tensor).view(1, 1, -1)
        if self.model_type == 'rnn':
            return self.rnn(embedded_tensor, hidden_tensor)
        if self.model_type == 'gru':
            return self.gru(embedded_tensor, hidden_tensor)
        if self.model_type == 'lstm':
            return self.lstm(embedded_tensor, hidden_tensor)

    def init_hidden(self):
        """Initialize hidden layers."""
        return torch.zeros(1, 1, self.hidden_size, device=const.DEVICE)
