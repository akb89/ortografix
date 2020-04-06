"""Constants used by ortografix."""

import torch

__all__ = ('EOS', 'SOS', 'SPACE', 'DEVICE')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EOS = '__EOS__'
SOS = '__SOS__'
SPACE = '__SPACE__'
