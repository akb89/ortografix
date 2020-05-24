"""Constants used by ortografix."""

import torch

__all__ = ('EOS', 'SOS', 'SEP', 'DEVICE', 'SOS_IDX', 'EOS_IDX', 'SEP_IDX')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOS = '__EOS__'
SOS = '__SOS__'
SEP = '__SEP__'
UNK = '__UNK__'
SOS_IDX = 0
EOS_IDX = 1
SEP_IDX = 2
