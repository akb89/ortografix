"""Preprocess data for training the seq2seq model."""

import random
import logging
import torch

import ortografix.utils.constants as const

logger = logging.getLogger(__name__)

__all__ = ('')


def prepare_batches(source_target_indexes, shuffle, batch_size):
    if shuffle:
        random.shuffle(source_target_indexes)
    for source_indexes, target_indexes in source_target_indexes:
        source_tensor = torch.tensor(source_indexes, dtype=torch.long,
                                     device=const.DEVICE).view(-1, 1)
        target_tensor = torch.tensor(target_indexes, dtype=torch.long,
                                     device=const.DEVICE).view(-1, 1)
    # TODO: stack tensors by batch


def prepare_token_based_source_target_indexes(input_stream, source_dict,
                                              target_dict, max_seq_len):
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
        source_tokens.preprend(const.SOS)
        source_tokens.append(const.EOS)
        target_tokens.preprend(const.SOS)
        target_tokens.append(const.EOS)
        source_indexes = [source_dict[token] for token in source_tokens]
        target_indexes = [target_dict[token] for token in target_tokens]
        source_target_indexes.append((source_indexes, target_indexes))
    logger.info('Source-target indexes contain {} pairs of items'
                .format(len(source_target_indexes)))
    return source_target_indexes


def prepare_character_based_source_target_indexes(input_stream, source_dict,
                                                  target_dict, max_seq_len):
    logger.info('Preparing character-based source-target indexes...')
    source_target_indexes = []
    for line in input_stream:
        line = line.strip()
        source_sent = line.split('\t')[0]
        target_sent = line.split('\t')[1]
        source_tokens = source_sent.split()
        target_tokens = target_sent.split()
        source_tokens.preprend(const.SOS)
        source_tokens.append(const.EOS)
        target_tokens.preprend(const.SOS)
        target_tokens.append(const.EOS)
        if len(source_tokens) != len(target_tokens):
            raise Exception(
                'In character_based mode source and target sentence '
                'pairs should contain the same number of tokens')
        if source_tokens[0] != const.SOS or target_tokens[0] != const.SOS:
            raise Exception(
                'Something went wrong: SOS should have been added to the '
                'beginning of all sentences')
        if source_tokens[-1] != const.EOS or target_tokens[-1] != const.EOS:
            raise Exception(
                'Something went wrong: EOS should have been added to the '
                'end of all sentences')
        for source_token, target_token in zip(source_tokens,
                                              target_tokens):
            source_indexes = []
            target_indexes = []
            if source_token in [const.SOS, const.EOS]:
                if source_token == const.EOS:
                    source_indexes.pop()
                    target_indexes.pop()  # remove last superfluous SPACE
                source_indexes.append(source_dict[source_token])
                target_indexes.append(target_dict[target_token])
            elif len(source_token) > max_seq_len or len(target_token) > max_seq_len:
                # careful! This can remove words in the middle of a sentence
                continue
            else:
                for source_char in source_token:
                    source_indexes.append(source_dict[source_char])
                for target_char in target_token:
                    target_indexes.append(target_dict[target_char])
                source_indexes.append(source_dict[const.SPACE])
                target_indexes.append(target_dict[const.SPACE])
            source_target_indexes.append((source_indexes, target_indexes))
    logger.info('Source-target indexes contain {} pairs of items'
                .format(len(source_target_indexes)))
    return source_target_indexes


def prepare_source_target_indexes(data_filepath, source_dict, target_dict,
                                  character_based, max_seq_len):
    with open(data_filepath, 'r', encoding='utf-8') as input_stream:
        if character_based:
            return prepare_character_based_source_target_indexes(
                input_stream, source_dict, target_dict, max_seq_len)
        return prepare_token_based_source_target_indexes(
            input_stream, source_dict, target_dict, max_seq_len)


def _prepare_source_target_dict(input_stream, character_based):
    logger.info('Preparing source and target dictionaries...')
    source_dict = {const.SOS: 0, const.EOS: 1}
    target_dict = {const.SOS: 0, const.EOS: 1}
    if character_based:
        source_dict[const.SPACE] = 2
        target_dict[const.SPACE] = 2
    for line in input_stream:
        line = line.strip()
        source_sent = line.split('\t')[0]
        target_sent = line.split('\t')[1]
        for token in source_sent.split():
            if character_based:
                for char in token:
                    if char not in source_dict:
                        source_dict[char] = len(source_dict)
            else:
                if token not in source_dict:
                    source_dict[token] = len(source_dict)
        for token in target_sent.split():
            if character_based:
                for char in token:
                    if char not in target_dict:
                        target_dict[char] = len(target_dict)
            else:
                if token not in target_dict:
                    target_dict[token] = len(target_dict)
    logger.info('Source dictionary contains {} items'.format(len(source_dict)))
    logger.info('Target dictionary contains {} items'.format(len(target_dict)))
    return source_dict, target_dict


def prepare_source_target_dict(data_filepath, character_based):
    with open(data_filepath, 'r', encoding='utf-8') as input_stream:
        return _prepare_source_target_dict(input_stream, character_based)


def preprocess(data_filepath, character_based=False, batch_size=1,
               shuffle=True, max_seq_len=10):
    """Prepare input tensors.

    Prepare dictionaries for source and target items.
    Discretize input to indexes.
    Convert to tensors and concatenate by batch
    """
    source_dict, target_dict = prepare_source_target_dict(
        data_filepath, character_based)
    source_target_indexes = prepare_source_target_indexes(
        data_filepath, source_dict, target_dict, character_based, max_seq_len)
    prepare_batches(source_target_indexes, shuffle, batch_size)
