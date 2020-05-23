"""Dataset model."""
import os
import random
import logging

import ortografix.utils.processing as putils

__all__ = ('Dataset', 'Vocab')

logger = logging.getLogger(__name__)


class Vocab():
    """A Vocab class to process vocabularies for source/target sequences."""

    def __init__(self, pairs=None, vocab_filepath=None,
                 is_character_based=False, is_source=True, is_reversed=False,
                 min_count=0):
        """Initialize vocabulary."""
        if not (pairs or vocab_filepath):
            raise Exception('You must specify either pairs or '
                            'vocab_filepath to initialize the vocabulary')
        if pairs and vocab_filepath:
            raise Exception('You cannot specify *both* pairs AND '
                            'vocab_filepath to initialize the vocabulary')
        if pairs:
            self._vocab = putils.create_vocab(
                pairs, is_character_based, is_source, is_reversed, min_count)
        if vocab_filepath:
            self._vocab = putils.load_vocab(vocab_filepath)

    @property
    def size(self):
        """Return the number of items in the vocabulary."""
        return len(self._vocab)

    @property
    def item2idx(self):
        """Return token/character-to-idx mapping (dict)."""
        return self._vocab

    @property
    def idx2item(self):
        """Return idx-to-token/character mapping (dict)."""
        return {idx: item for item, idx in self._vocab.items()}


# pylint: disable=R0902
class Dataset():
    """A dataset class to return source/target tensors from training data."""

    def __init__(self, pairs, is_character_based, shuffle,
                 max_seq_len, is_reversed, min_count):
        """Prepare input tensors.

        Prepare dictionaries for source and target items.
        Discretize input to indexes.
        Convert to tensors and concatenate by batch
        """
        self._pairs = pairs
        self._is_character_based = is_character_based
        self._shuffle = shuffle
        self._is_reversed = is_reversed
        if not max_seq_len:
            self._max_seq_len = putils.get_max_seq_len(
                pairs, is_character_based)
            logger.info('max_seq_len not specified in args. Automatically setting '
                        'to longest source sequence length = {}'
                        .format(max_seq_len))
        else:
            logger.info('Manually setting max_seq_len to {}'
                        .format(max_seq_len))
            self._max_seq_len = max_seq_len
        self._source_vocab = Vocab(pairs=pairs,
                                   is_character_based=is_character_based,
                                   is_source=True, is_reversed=is_reversed,
                                   min_count=min_count)
        self._target_vocab = Vocab(pairs=pairs,
                                   is_character_based=is_character_based,
                                   is_source=False, is_reversed=is_reversed,
                                   min_count=min_count)
        self._indexes = putils.index_pairs(
            self._pairs, self._source_vocab.item2idx,
            self._target_vocab.item2idx, self._is_character_based,
            self._max_seq_len, self._is_reversed)

    @property
    def is_character_based(self):
        """Return True if character-based, False if token-based."""
        return self._is_character_based

    @property
    def shuffle(self):
        """Return True if dataset sentence pairs will be shuffled."""
        return self._shuffle

    @property
    def max_seq_len(self):
        """Return the maximal sequence length to be considered."""
        return self._max_seq_len

    @property
    def is_reversed(self):
        """Return whether or not second item should be considered source."""
        return self._is_reversed

    @property
    def source_vocab(self):
        """Return vocabulary of source."""
        return self._source_vocab

    @property
    def target_vocab(self):
        """Return vocabulary of target."""
        return self._target_vocab

    @property
    def indexes(self):
        """Return a list of source/target indexed sequences."""
        if self._shuffle:
            random.shuffle(self._indexes)
        return self._indexes

    def save_params(self, output_dirpath):
        """Save vocabularies and dataset parameters."""
        logger.info('Saving dataset to directory {}'.format(output_dirpath))
        params_filepath = os.path.join(output_dirpath, 'dataset.params')
        with open(params_filepath, 'w', encoding='utf-8') as output_str:
            print('shuffle\t{}'.format(self._shuffle), file=output_str)
            print('is_character_based\t{}'.format(self._is_character_based),
                  file=output_str)
            print('max_seq_len\t{}'.format(self._max_seq_len), file=output_str)
            print('is_reversed\t{}'.format(self._is_reversed), file=output_str)
        logger.info('Saving source vocab...')
        source_vocab_filepath = os.path.join(output_dirpath, 'source.vocab')
        with open(source_vocab_filepath, 'w', encoding='utf-8') as source_str:
            for item, idx in self._source_vocab.item2idx.items():
                print('{}\t{}'.format(item, idx), file=source_str)
        logger.info('Saving target vocab...')
        target_vocab_filepath = os.path.join(output_dirpath, 'target.vocab')
        with open(target_vocab_filepath, 'w', encoding='utf-8') as target_str:
            for item, idx in self._target_vocab.item2idx.items():
                print('{}\t{}'.format(item, idx), file=target_str)
