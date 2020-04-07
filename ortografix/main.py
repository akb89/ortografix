"""Welcome to ortografix.

This is the entry point of the application.
"""
import os

import argparse
import random
import logging
import logging.config

import torch
from torch import optim

import ortografix.utils.config as cutils
import ortografix.utils.constants as const

from ortografix.model.encoder import Encoder
from ortografix.model.decoder import Decoder
from ortografix.model.dataset import Dataset


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('train', 'decode', 'evaluate')


def _train_single_batch(source_tensor, target_tensor, encoder, decoder,
                        encoder_optimizer, decoder_optimizer, max_seq_len,
                        criterion, use_teacher_forcing, teacher_forcing_ratio):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_seq_len, encoder.hidden_size,
                                  device=const.DEVICE)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[const.SOS_idx]], device=const.DEVICE)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def _train(encoder, decoder, dataset, num_epochs, learning_rate, print_every):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    num_iter = 0
    num_total_iters = len(dataset.source_target_indexes) * num_epochs
    for epoch in range(1, num_epochs+1):
        for source_tensor, target_tensor in dataset.prepare_source_target_tensors():
            num_iter += 1
            loss = _train_single_batch(
                source_tensor, target_tensor, encoder, decoder,
                encoder_optimizer, decoder_optimizer, dataset.max_seq_len,
                criterion, use_teacher_forcing, teacher_forcing_ratio)
            print_loss_total = loss
            if num_iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                time_info = tutils.time_since(start,
                                              num_iter / num_total_iters)
                logger.info('{} ({} {}%) {}'
                            .format(time_info, num_iter,
                                    num_iter / num_total_iters * 100,
                                    print_loss_avg))


def _decode():
    pass


def train(args):
    """Train the model."""
    logger.info('Training model from {}'.format(args.data))
    dataset = Dataset(args.data, args.character_based, args.shuffle,
                      args.max_seq_len)
    encoder = Encoder(model_type=args.model_type,
                      input_size=dataset.source_vocab_size,
                      hidden_size=args.hidden_size,
                      num_layers=args.num_layers,
                      nonlinearity='args.nonlinearity,
                      bias=args.bias, batch_first=args.batch_first,
                      dropout=args.dropout,
                      bidirectional=args.bidirectional).to(const.DEVICE)
    decoder = Decoder(model_type=args.model_type,
                      hidden_size=args.hidden_size,
                      output_size=dataset.target_vocab_size,
                      num_layers=args.num_layers,
                      nonlinearity='args.nonlinearity,
                      bias=args.bias, batch_first=args.batch_first,
                      dropout=args.dropout,
                      bidirectional=args.bidirectional).to(const.DEVICE)
    return _train(encoder, decoder, dataset, args.learning_rate)


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
