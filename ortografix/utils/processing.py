"""Process data and params for training/decoding."""
import sys
import logging

from collections import defaultdict

import ortografix.utils.constants as const

logger = logging.getLogger(__name__)

__all__ = ('index_pairs', 'index_sequence', 'create_vocab', 'load_vocab',
           'load_params', 'get_max_seq_len')


def get_max_seq_len(pairs, is_character_based):
    """Return max sequence length computed from word or sentencep pairs.

    If is_character_based, will return the length of the longuest word.
    Else, will return the length of the longuest sentence.
    """
    max_seq_len = 0
    for pair in pairs:
        tokens = pair[0].split()
        if is_character_based:
            for token in tokens:
                max_seq_len = max(max_seq_len, len(token))
        else:
            max_seq_len = max(max_seq_len, len(tokens))
        tokens = pair[1].split()
        if is_character_based:
            for token in tokens:
                max_seq_len = max(max_seq_len, len(token))
        else:
            max_seq_len = max(max_seq_len, len(tokens))
    return max_seq_len


def load_params(params_filepath):
    """Load parameter dict from file (tab separated)."""
    params = {}
    with open(params_filepath, 'r', encoding='utf-8') as params_str:
        for line in params_str:
            line = line.strip()
            items = line.split('\t')
            if items[0] in ['shuffle', 'is_character_based', 'cuda',
                            'is_reversed']:
                params[items[0]] = items[1] == 'True'
            elif items[0] in ['max_seq_len']:
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


def _count_items(pairs, is_character_based, is_source, is_reversed):
    counts = defaultdict(int)
    ref_idx = 1 if is_source == is_reversed else 0
    for pair in pairs:
        sent = pair[ref_idx]
        for token in sent.split():
            if is_character_based is True:
                for char in token:
                    counts[char] += 1
            elif is_character_based is False:
                counts[token] += 1
            else:
                raise Exception('Unsupported value {} for is_character_based '
                                'parameter'.format(is_character_based))
    return counts


def create_vocab(pairs, is_character_based, is_source, is_reversed, min_count):
    """Generate token/character-to-idx mapping for word or sent pairs.

    source_or_target should specify whether vocab should be created for source
    or target sequence
    """
    logger.info('Preparing source and target dictionaries...')
    vocab = {const.SOS: const.SOS_IDX, const.EOS: const.EOS_IDX}
    ref_idx = 1 if is_source == is_reversed else 0
    counts = _count_items(pairs, is_character_based, is_source, is_reversed)
    for pair in pairs:
        sent = pair[ref_idx]
        for token in sent.split():
            if is_character_based is True:
                for char in token:
                    if char not in vocab and counts[char] >= min_count:
                        vocab[char] = len(vocab)
            elif is_character_based is False:
                if token not in vocab and counts[token] >= min_count:
                    vocab[token] = len(vocab)
            else:
                raise Exception('Unsupported value {} for is_character_based '
                                'parameter'.format(is_character_based))
    vocab[const.UNK] = len(vocab)
    desc = 'Source' if is_source else 'Target'
    logger.info('{} vocabulary contains {} items'.format(desc, len(vocab)))
    return vocab


def index_sequence(sequence, item2idx, is_character_based):
    """Convert a sequence to a list of int following item2idx.

    TODO: add support for OOV.
    """
    indexes = [const.SOS]
    sequence = sequence.strip()
    for item in sequence.split():
        if is_character_based:
            raise Exception('Unsupported is_character_based option for now')
        indexes.append(item2idx[item])
    indexes.append(const.EOS)
    return indexes


def _index_tokens(pairs, source_item2idx, target_item2idx, max_seq_len,
                  is_reversed):
    logger.info('Preparing token-based source-target indexes...')
    source_target_indexes = []
    for pair in pairs:
        source_sent = pair[1] if is_reversed else pair[0]
        target_sent = pair[0] if is_reversed else pair[1]
        source_tokens = source_sent.split()
        target_tokens = target_sent.split()
        if len(source_tokens) > max_seq_len \
         or len(target_tokens) > max_seq_len:
            continue
        source_tokens = [const.SOS] + source_tokens  # prepend value
        source_tokens.append(const.EOS)
        target_tokens = [const.SOS] + target_tokens  # prepend value
        target_tokens.append(const.EOS)
        source_indexes = [source_item2idx[token] for token in source_tokens]
        target_indexes = [target_item2idx[token] for token in target_tokens]
        source_target_indexes.append((source_indexes, target_indexes))
    logger.info('Source-target indexes contain {} pairs of items'
                .format(len(source_target_indexes)))
    return source_target_indexes


# pylint: disable=R0914
def _index_characters(pairs, source_item2idx, target_item2idx,
                      max_seq_len, is_reversed):
    logger.info('Preparing character-based source-target indexes...')
    source_target_indexes = []
    for pair in pairs:
        source_sent = pair[1] if is_reversed else pair[0]
        target_sent = pair[0] if is_reversed else pair[1]
        source_tokens = source_sent.split()
        target_tokens = target_sent.split()
        for source_token, target_token in zip(source_tokens,
                                              target_tokens):
            source_indexes = []
            target_indexes = []
            if len(source_token) > max_seq_len \
             or len(target_token) > max_seq_len:
                # careful! This can remove words in the middle of a sentence
                continue
            # Here, string sequence is taken to be word/token. SOS indicates
            # the start of a token, EOS its end.
            source_indexes.append(source_item2idx[const.SOS])
            target_indexes.append(target_item2idx[const.SOS])
            for source_char in source_token:
                if source_char not in source_item2idx:
                    source_indexes.append(source_item2idx[const.UNK])
                else:
                    source_indexes.append(source_item2idx[source_char])
            for target_char in target_token:
                if target_char not in target_item2idx:
                    target_indexes.append(target_item2idx[const.UNK])
                else:
                    target_indexes.append(target_item2idx[target_char])
            source_indexes.append(source_item2idx[const.EOS])
            target_indexes.append(target_item2idx[const.EOS])
            source_target_indexes.append((source_indexes, target_indexes))
    logger.info('Source-target indexes contain {} pairs of items'
                .format(len(source_target_indexes)))
    return source_target_indexes


def _is_dataset_consistent(data_filepath):
    """Check that all lines contains the same number of tokens.

    This is necessary only for character-based indexing.
    """
    is_consistent = True
    with open(data_filepath, 'r', encoding='utf-8') as data_str:
        for line in data_str:
            line = line.strip()
            source_sent = line.split('\t')[0]
            target_sent = line.split('\t')[1]
            source_tokens = source_sent.split()
            target_tokens = target_sent.split()
            if len(source_tokens) != len(target_tokens):
                is_consistent = False
                logger.error('Source and target sequences do NOT contain same '
                             'number of tokens in: \n source: {}\n target: {}'
                             .format(source_sent, target_sent))
    return is_consistent


def index_pairs(pairs, source_item2idx, target_item2idx,
                is_character_based, max_seq_len, is_reversed):
    """Convert input data text file to list of indexed integers."""
    logger.info('Indexing pairs...')
    # if is_character_based:
    #     if not _is_dataset_consistent(data_filepath):
    #         logger.error('Dataset in inconsistent state. Exiting...')
    #         sys.exit(0)
    if is_character_based:
        return _index_characters(pairs, source_item2idx,
                                 target_item2idx, max_seq_len, is_reversed)
    return _index_tokens(pairs, source_item2idx, target_item2idx,
                         max_seq_len, is_reversed)
