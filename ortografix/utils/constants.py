"""Constants used by ortografix."""

import torch

__all__ = ('EOS', 'SOS', 'SEP', 'DEVICE', 'SOS_idx', 'EOS_idx')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOS = '__EOS__'
SOS = '__SOS__'
SEP = '__SEP__'
SOS_idx = 0
EOS_idx = 1
