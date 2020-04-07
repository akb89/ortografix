"""Preprocessing utilities."""
import random

import torch

import ortografix.utils.preprocessing as putils
import ortografix.utils.constants as const

__all__ = ('Dataset')


class Dataset():
    """A dataset class to return source/target tensors from training data."""

    def __init__(self, data_filepath, character_based=False, shuffle=True,
                 max_seq_len=10):
        """Prepare input tensors.

        Prepare dictionaries for source and target items.
        Discretize input to indexes.
        Convert to tensors and concatenate by batch
        """
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        source_dict, target_dict = putils.prepare_source_target_dict(
            data_filepath, character_based)
        # self.source_dict = source_dict
        # self.target_dict = target_dict
        self.source_vocab_size = len(source_dict)
        self.target_vocab_size = len(target_dict)
        self.source_target_indexes = putils.prepare_source_target_indexes(
            data_filepath, source_dict, target_dict, character_based,
            max_seq_len)

    def prepare_source_target_tensors(self):
        if self.shuffle:
            random.shuffle(self.source_target_indexes)
        for source_indexes, target_indexes in self.source_target_indexes:
            source_tensor = torch.tensor(source_indexes, dtype=torch.long,
                                         device=const.DEVICE).view(-1, 1)
            target_tensor = torch.tensor(target_indexes, dtype=torch.long,
                                         device=const.DEVICE).view(-1, 1)
            yield source_tensor, target_tensor
