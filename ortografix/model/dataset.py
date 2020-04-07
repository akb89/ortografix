"""Preprocessing utilities."""
import os
import random
import logging

import torch

import ortografix.utils.preprocessing as putils
import ortografix.utils.constants as const

__all__ = ('Dataset')

logger = logging.getLogger(__name__)


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
        self.source_dict, self.target_dict = putils.prepare_source_target_dict(
            data_filepath, character_based)
        self.source_vocab_size = len(self.source_dict)
        self.target_vocab_size = len(self.target_dict)
        self.source_target_indexes = putils.prepare_indexed_pairs(
            data_filepath, self.source_dict, self.target_dict, character_based,
            max_seq_len)

    def input_tensors(self):
        """Return iterator over source/target pairs of tensors."""
        if self.shuffle:
            random.shuffle(self.source_target_indexes)
        for source_indexes, target_indexes in self.source_target_indexes:
            source_tensor = torch.tensor(source_indexes, dtype=torch.long,
                                         device=const.DEVICE).view(-1, 1)
            target_tensor = torch.tensor(target_indexes, dtype=torch.long,
                                         device=const.DEVICE).view(-1, 1)
            yield source_tensor, target_tensor

    def save_params(self, output_dirpath):
        """Save vocabularies and dataset parameters."""
        logger.info('Saving dataset to directory {}'.format(output_dirpath))
        params_filepath = os.path.join(output_dirpath, 'dataset.params')
        with open(params_filepath, 'w', encoding='utf-8') as output_str:
            print('shuffle\t{}'.format(self.shuffle), file=output_str)
            print('max_seq_len\t{}'.format(self.max_seq_len), file=output_str)
        logger.info('Saving source vocab...')
        source_vocab_filepath = os.path.join(output_dirpath, 'source.vocab')
        with open(source_vocab_filepath, 'w', encoding='utf-8') as source_str:
            for item, idx in self.source_dict.items():
                print('{}\t{}'.format(item, idx), file=source_str)
        logger.info('Saving target vocab...')
        target_vocab_filepath = os.path.join(output_dirpath, 'target.vocab')
        with open(target_vocab_filepath, 'w', encoding='utf-8') as target_str:
            for item, idx in self.target_dict.items():
                print('{}\t{}'.format(item, idx), file=target_str)
