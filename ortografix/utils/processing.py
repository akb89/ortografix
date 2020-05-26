"""Process data and params for training/decoding."""
import logging

from collections import defaultdict

import ortografix.utils.constants as const

logger = logging.getLogger(__name__)

__all__ = ('index_pairs', 'index_sequence', 'create_vocab', 'load_vocab',
           'load_params', 'get_max_seq_len')


def get_max_seq_len(indexed_pairs):
    """Return max sequence length computed from index pairs.

    indexes already include SOS/EOS/SEP.
    """
    max_seq_len = 0
    # max_seq = ''
    for indexed_pair in indexed_pairs:
        max_seq_len = max(max_seq_len, len(indexed_pair[0]))
        max_seq_len = max(max_seq_len, len(indexed_pair[1]))
        # if len(indexed_pair[0]) > max_seq_len:
        #     max_seq_len = len(indexed_pair[0])
        #     max_seq = indexed_pair[0]
        # if len(indexed_pair[1]) > max_seq_len:
        #     max_seq_len = len(indexed_pair[1])
        #     max_seq = indexed_pair[1]
    # print(max_seq_len, max_seq)
    return max_seq_len


def load_params(params_filepath):
    """Load parameter dict from file (tab separated)."""
    params = {}
    with open(params_filepath, 'r', encoding='utf-8') as params_str:
        for line in params_str:
            line = line.strip()
            items = line.split('\t')
            if items[0] in ['shuffle', 'cuda', 'reverse']:
                params[items[0]] = items[1] == 'True'
            elif items[0] in ['max_seq_len', 'min_count']:
                params[items[0]] = int(items[1])
            else:
                raise Exception('Unsupported dataset parameter: {}'
                                .format(items[0]))
    return params


def load_vocab(vocab_filepath):
    """Load previously saved vocab to dic."""
    vocab = {}
    with open(vocab_filepath, 'r', encoding='utf-8') as vocab_str:
        for line in vocab_str:
            line = line.strip()
            items = line.split('\t')
            vocab[items[0]] = int(items[1])
    return vocab


def count_chars(sequences):
    """Count characters in sequences."""
    counts = defaultdict(int)
    for seq in sequences:
        for token in seq:
            for char in token:
                counts[char] += 1
    return counts


def create_vocab(sequences, min_count, counts):
    """Generate character-to-idx mapping from list of sequences."""
    vocab = {const.SOS: const.SOS_IDX, const.EOS: const.EOS_IDX,
             const.SEP: const.SEP_IDX}
    for seq in sequences:
        for token in seq:
            for char in token:
                if char not in vocab and counts[char] >= min_count:
                    vocab[char] = len(vocab)
    vocab[const.UNK] = len(vocab)
    return vocab


def index_sequence(sequence, char2idx, sos_idx=const.SOS_IDX,
                   sep_idx=const.SEP_IDX, eos_idx=const.EOS_IDX,
                   unk=const.UNK):
    """Convert a sequence to a list of int following item2idx."""
    indexes = [sos_idx]
    for token in sequence:
        for char in token:
            if char in char2idx:
                indexes.append(char2idx[char])
            else:
                indexes.append(char2idx[unk])
        indexes.append(sep_idx)
    indexes.pop()
    indexes.append(eos_idx)
    return indexes


def index_pairs(pairs, left_char2idx, right_char2idx):
    """Index pairs.

    Discretize list of sequence pairs. Sequences can correspond to sentences
    or bare words. System will adjust accordingly.
    """
    indexes = []
    for pair in pairs:
        indexes.append((index_sequence(pair[0], left_char2idx),
                        index_sequence(pair[1], right_char2idx)))
    return indexes


def convert_to_seq_pairs(data_filepath):
    """Convert tab-separated sequences in text file to sequence pairs.

    Will check consistency: sequences must contain the same number of tokens.
    Will return a list of tuples of lists. Ex:
        This is a test \t This iz a test
        [([This, is, a, test], [This, iz, a, test])]

    """
    pairs = []
    logger.info('Processing file {}'.format(data_filepath))
    with open(data_filepath, 'r', encoding='utf-8') as input_str:
        for line in input_str:
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens[0].split()) != len(tokens[1].split()):
                raise Exception('Dataset in inconsistent state. Sequences '
                                'should contain same number of tokens in: {}'
                                .format(line))
            pairs.append((tokens[0].split(), tokens[1].split()))
    logger.info('Input file contains {} sequence pairs'.format(len(pairs)))
    return pairs
