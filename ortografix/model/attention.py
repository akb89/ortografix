"""Attention Decoder."""
import torch

import ortografix.utils.constants as const

__all__ = ('Attention')


# pylint: disable=R0902
class Attention(torch.nn.Module):
    """Attention class."""

    def __init__(self, hidden_size, output_size, max_seq_len, num_layers,
                 nonlinearity, bias, dropout):
        """Initialize attention model."""
        super(Attention, self).__init__()
        self.model_type = 'gru'
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.with_attention = True
        self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_seq_len)
        # if bidirectional:
        #     self.attn_combine = torch.nn.Linear(self.hidden_size * 3,
        #                                         self.hidden_size)
        # else:
        #     self.attn_combine = torch.nn.Linear(self.hidden_size * 2,
        #                                         self.hidden_size)
        self.attn_combine = torch.nn.Linear(self.hidden_size * 2,
                                            self.hidden_size)
        self.gru = torch.nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, bias=bias, batch_first=False,
            dropout=dropout, bidirectional=False)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    # pylint: disable=R1710, W0221
    def forward(self, input_tensor, hidden, encoder_outputs):
        """Apply forward computation."""
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        if self.model_type == 'lstm':
            attn_weights = torch.nn.functional.softmax(
                self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        else:
            attn_weights = torch.nn.functional.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = torch.nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights

    # def init_hidden(self):
    #     """Initialize hidden layers."""
    #     return torch.zeros(self.num_layers, 1, self.hidden_size,
    #                        device=const.DEVICE)
