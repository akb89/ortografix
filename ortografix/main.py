"""Welcome to ortografix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import ortografix.utils.config as cutils


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('train', 'decode', 'evaluate')


def _train():
    pass


def _decode():
    pass


def train(args):
    """Train the model."""
    logger.info('Training model from {}'.format(args.data))
    return _train()


def decode(args):
    """Decode the input."""
    logger.info('Decoding input from {}'.format(args.data))
    return _decode()


def evaluate(args):
    """Evaluate a given model on a test set."""
    pass


def main():
    """Launch ortografix."""
    parser = argparse.ArgumentParser(prog='ortografix')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        help='train')
    parser_train.set_defaults(func=train)
    # parser_extract.add_argument('-v', '--vectors', required=True,
    #                             help='input vectors in txt format')
    parser_decode = subparsers.add_parser(
        'decode', formatter_class=argparse.RawTextHelpFormatter,
        help='decode input string: convert to simplified ortography')
    parser_decode.set_defaults(func=decode)
    # parser_convert.add_argument('-w', '--what', required=True,
    #                             choices=['bert', 'numpy'],
    #                             help='absolute path to vocabulary')
    # parser_convert.add_argument('-v', '--vocab', required=True,
    #                             help='absolute path to vocabulary')
    # parser_convert.add_argument('-m', '--model',
    #                             help='absolute path to numpy model')
    # parser_reduce = subparsers.add_parser(
    #     'reduce', formatter_class=argparse.RawTextHelpFormatter,
    #     help='align numpy model vocabularies.')
    # parser_reduce.set_defaults(func=_align_vocabs_and_models)
    # parser_reduce.add_argument('-i', '--model-dir', required=True,
    #                            help='absolute path to .npy models '
    #                                 'directory. The directory should '
    #                                 'contain the .vocab file '
    #                                 'corresponding to the .npy model.')
    # parser_reduce.add_argument('-t', '--export-to-text',
    #                            action='store_true',
    #                            help='if passed, will also export models'
    #                                 'to .text format')
    args = parser.parse_args()
    args.func(args)
