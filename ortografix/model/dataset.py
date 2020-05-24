"""Dataset model."""
import os
import random
import logging

import ortografix.utils.processing as putils

__all__ = ('Dataset', 'Vocab')

logger = logging.getLogger(__name__)


class Vocab():
    """A Vocab class to process vocabularies for source/target sequences."""

    def __init__(self, sequences=None, vocab_filepath=None, min_count=0):
        """Initialize vocabulary."""
        if not (sequences or vocab_filepath):
            raise Exception('You must specify either pairs or '
                            'vocab_filepath to initialize the vocabulary')
        if sequences and vocab_filepath:
            raise Exception('You cannot specify *both* pairs AND '
                            'vocab_filepath to initialize the vocabulary')
        if sequences:
            self._counts = putils.count_chars(sequences)
            self._vocab = putils.create_vocab(sequences, min_count,
                                              self._counts)
        if vocab_filepath:
            self._vocab = putils.load_vocab(vocab_filepath)

    @property
    def size(self):
        """Return the number of items in the vocabulary."""
        return len(self._vocab)

    @property
    def counts(self):
        """Return the vocabulary items counts."""
        return len(self._counts)

    @property
    def char2idx(self):
        """Return char-to-idx mapping (dict)."""
        return self._vocab

    @property
    def idx2char(self):
        """Return idx-to-char mapping (dict)."""
        return {idx: char for char, idx in self._vocab.items()}


# pylint: disable=R0902
class Dataset():
    """A dataset class to return source/target tensors from training data."""

    def __init__(self, pairs, shuffle, max_seq_len, min_count, reverse):
        """Prepare input tensors.

        Prepare dictionaries for source and target items.
        Discretize input to indexes.
        Convert to tensors and concatenate by batch
        """
        self._shuffle = shuffle
        self._reverse = reverse
        self._min_count = min_count
        self._left_vocab = Vocab(sequences=[pair[0] for pair in pairs],
                                 min_count=min_count)
        self._right_vocab = Vocab(sequences=[pair[1] for pair in pairs],
                                  min_count=min_count)
        self._indexed_pairs = putils.index_pairs(
            pairs, self._left_vocab.char2idx, self._right_vocab.char2idx)
        if not max_seq_len:
            self._max_seq_len = putils.get_max_seq_len(self._indexed_pairs)
            logger.info(
                'max_seq_len not specified in args. Automatically setting to '
                'longest source sequence length = {}'
                .format(self._max_seq_len))
        else:
            self._max_seq_len = max_seq_len
            logger.info('Manually setting max_seq_len to {}'
                        .format(self._max_seq_len))

    @property
    def shuffle(self):
        """Return True if dataset sequence pairs will be shuffled."""
        return self._shuffle

    @property
    def reverse(self):
        """Return True if sequence pairs should be reversed.

        That is, return True if source/target is right/left instead of
        left/right in the input sequence pair.
        """
        return self._reverse

    @property
    def max_seq_len(self):
        """Return the maximal sequence length to be considered."""
        return self._max_seq_len

    @property
    def right_vocab(self):
        """Return vocabulary corresponding to rightward elements in seq."""
        return self._right_vocab

    @property
    def left_vocab(self):
        """Return vocabulary corresponding to leftward elements in seq."""
        return self._left_vocab

    @property
    def indexed_pairs(self):
        """Return a list of source/target indexed sequences."""
        if self._shuffle:
            random.shuffle(self._indexed_pairs)
        return self._indexed_pairs

    def save_params(self, output_dirpath):
        """Save vocabularies and dataset parameters."""
        logger.info('Saving dataset to directory {}'.format(output_dirpath))
        params_filepath = os.path.join(output_dirpath, 'dataset.params')
        with open(params_filepath, 'w', encoding='utf-8') as output_str:
            print('shuffle\t{}'.format(self._shuffle), file=output_str)
            print('max_seq_len\t{}'.format(self._max_seq_len), file=output_str)
            print('min_count\t{}'.format(self._min_count), file=output_str)
            print('reverse\t{}'.format(self._reverse), file=output_str)
        left_vocab_filepath = os.path.join(output_dirpath, 'left.vocab')
        with open(left_vocab_filepath, 'w', encoding='utf-8') as left_str:
            for char, idx in self._left_vocab.char2idx.items():
                print('{}\t{}'.format(char, idx), file=left_str)
        right_vocab_filepath = os.path.join(output_dirpath, 'right.vocab')
        with open(right_vocab_filepath, 'w', encoding='utf-8') as right_str:
            for char, idx in self._right_vocab.char2idx.items():
                print('{}\t{}'.format(char, idx), file=right_str)
