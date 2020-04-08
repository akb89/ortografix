"""Preprocess data for training the seq2seq model."""
import sys
import logging

import ortografix.utils.constants as const

logger = logging.getLogger(__name__)

__all__ = ('index_dataset', 'index_sequence', 'create_vocab', 'load_vocab')


def load_vocab(vocab_filepath):
    """Load previously saved vocab to dic."""
    vocab = {}
    with open(vocab_filepath, 'r', encoding='utf-8') as vocab_str:
        for line in vocab_str:
            line = line.strip()
            items = line.split('\t')
            vocab[items[0]] = int(items[1])

    return vocab


def _create_vocab(input_str, is_character_based, is_source):
    logger.info('Preparing source and target dictionaries...')
    vocab = {const.SOS: const.SOS_idx, const.EOS: const.EOS_idx}
    ref_idx = 0 if is_source is True else 1
    for line in input_str:
        line = line.strip()
        sent = line.split('\t')[ref_idx]
        for token in sent.split():
            if is_character_based is True:
                for char in token:
                    if char not in vocab:
                        vocab[char] = len(vocab)
            elif is_character_based is False:
                if token not in vocab:
                    vocab[token] = len(vocab)
            else:
                raise Exception('Unsupported value {} for is_character_based '
                                'parameter'.format(is_character_based))
    desc = 'Source' if is_source else 'Target'
    logger.info('{} vocabulary contains {} items'.format(desc, len(vocab)))
    return vocab


def create_vocab(dataset_filepath, character_based, source_or_target):
    """Generate token/character-to-idx mapping for a dataset file.

    Dataset file should contains pairs of aligned sentences,
    even in character-based mode.
    source_or_target should specify whether vocab should be created for source
    or target sequence
    """
    with open(dataset_filepath, 'r', encoding='utf-8') as input_str:
        return _create_vocab(input_str, character_based, source_or_target)


def index_sequence(sequence, item2idx, is_character_based):
    """Convert a sequence to a list of int following item2idx.

    TODO: add support for OOV.
    """
    indexes = [const.SOS]
    sequence = sequence.strip()
    for item in sequence.split():
        if is_character_based:
            raise Exception('Unsupported for now')
        else:
            indexes.append(item2idx[item])
    indexes.append(const.EOS)
    return indexes


def _index_tokens(input_stream, source_item2idx, target_item2idx, max_seq_len):
    logger.info('Preparing token-based source-target indexes...')
    source_target_indexes = []
    for line in input_stream:
        line = line.strip()
        source_sent = line.split('\t')[0]
        target_sent = line.split('\t')[1]
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


def _index_characters(input_stream, source_item2idx, target_item2idx,
                      max_seq_len):
    logger.info('Preparing character-based source-target indexes...')
    source_target_indexes = []
    for line in input_stream:
        line = line.strip()
        source_sent = line.split('\t')[0]
        target_sent = line.split('\t')[1]
        source_tokens = source_sent.split()
        target_tokens = target_sent.split()
        for source_token, target_token in zip(source_tokens,
                                              target_tokens):
            source_indexes = []
            target_indexes = []
            if len(source_token) > max_seq_len or len(target_token) > max_seq_len:
                # careful! This can remove words in the middle of a sentence
                continue
            # Here, string sequence is taken to be word/token. SOS indicates
            # the start of a token, EOS its end.
            source_indexes.append(source_item2idx[const.SOS])
            target_indexes.append(target_item2idx[const.SOS])
            for source_char in source_token:
                source_indexes.append(source_item2idx[source_char])
            for target_char in target_token:
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


def index_dataset(data_filepath, source_item2idx, target_item2idx,
                  is_character_based, max_seq_len):
    logger.info('Indexing dataset...')
    if is_character_based:
        if not _is_dataset_consistent(data_filepath):
            logger.error('Dataset in inconsistent state. Exiting...')
            sys.exit(0)
    with open(data_filepath, 'r', encoding='utf-8') as data_str:
        if is_character_based:
            return _index_characters(data_str, source_item2idx,
                                     target_item2idx, max_seq_len)
        return _index_tokens(data_str, source_item2idx, target_item2idx,
                             max_seq_len)
