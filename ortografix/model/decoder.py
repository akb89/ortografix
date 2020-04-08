"""Simple Decoder without Attention."""
import logging

import torch

import ortografix.utils.constants as const

__all__ = ('Decoder')

logger = logging.getLogger(__name__)


# pylint: disable=R0902
class Decoder(torch.nn.Module):
    """Decoder class."""

    def __init__(self, model_type, hidden_size, output_size, num_layers=1,
                 nonlinearity='tanh', bias=True, dropout=0,
                 bidirectional=False):
        """Initialize decoder model."""
        if model_type not in ['rnn', 'gru', 'lstm']:
            raise Exception('Unsupported model type: {}'.format(model_type))
        super(Decoder, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
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
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    # pylint: disable=R1710, W0221
    def forward(self, input_tensor, hidden_tensor):
        """Apply forward computation."""
        output_tensor = self.embedding(input_tensor).view(1, 1, -1)
        output_tensor = torch.nn.functional.relu(output_tensor)
        if self.model_type == 'rnn':
            output_tensor, hidden_tensor = self.rnn(output_tensor,
                                                    hidden_tensor)
        elif self.model_type == 'gru':
            output_tensor, hidden_tensor = self.gru(output_tensor,
                                                    hidden_tensor)
        else:
            output_tensor, hidden_tensor = self.lstm(output_tensor,
                                                     hidden_tensor)
        output_tensor = self.softmax(self.out(output_tensor[0]))
        return output_tensor, hidden_tensor

    def init_hidden(self):
        """Initialize hidden layers."""
        return torch.zeros(1, 1, self.hidden_size, device=const.DEVICE)
