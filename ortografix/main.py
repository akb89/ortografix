"""Welcome to ortografix.

This is the entry point of the application.
"""
import os

import argparse
import random
import time
import logging
import logging.config

import torch
from torch import optim

import ortografix.utils.config as cutils
import ortografix.utils.constants as const
import ortografix.utils.time as tutils

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
    # add 2 to max_seq_len to include SOS and EOS
    encoder_outputs = torch.zeros(max_seq_len+2, encoder.hidden_size,
                                  device=const.DEVICE)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(source_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[const.SOS_idx]], device=const.DEVICE)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == const.EOS_idx:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def _train(encoder, decoder, dataset, num_epochs, learning_rate, print_every,
           use_teacher_forcing, teacher_forcing_ratio):
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
                logger.info('Epoch {}/{} {} ({} {}%) {}'
                            .format(epoch, num_epochs, time_info, num_iter,
                                    round(num_iter / num_total_iters * 100),
                                    round(print_loss_avg, 4)))


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
                      nonlinearity=args.nonlinearity,
                      bias=args.bias, dropout=args.dropout,
                      bidirectional=args.bidirectional).to(const.DEVICE)
    decoder = Decoder(model_type=args.model_type,
                      hidden_size=args.hidden_size,
                      output_size=dataset.target_vocab_size,
                      num_layers=args.num_layers,
                      nonlinearity=args.nonlinearity,
                      bias=args.bias, dropout=args.dropout,
                      bidirectional=args.bidirectional).to(const.DEVICE)
    return _train(encoder, decoder, dataset, args.epochs, args.learning_rate,
                  args.print_every, args.use_teacher_forcing,
                  args.teacher_forcing_ratio)


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
        help='train the seq2seq model')
    parser_train.set_defaults(func=train)
    parser_train.add_argument('-d', '--data', required=True,
                              help='absolute path to training data')
    parser_train.add_argument('-t', '--model-type',
                              choices=['rnn', 'gru', 'lstm'],
                              default='gru',
                              help='encoder/decoder model type')
    parser_train.add_argument('-c', '--character-based', action='store_true',
                              help='if set, will switch from token-based to '
                                   'character-based model. To be used only '
                                   'for ortographic simplification, not '
                                   'neural machine translation')
    parser_train.add_argument('-s', '--shuffle', action='store_true',
                              help='if set, will shuffle the training data')
    parser_train.add_argument('-m', '--max-seq-len', type=int, default=10,
                              help='maximum sequence length to retain')
    parser_train.add_argument('-z', '--hidden-size', type=int, default=256,
                              help='size of the hidden layer')
    parser_train.add_argument('-n', '--num-layers', type=int, default=1,
                              help='number of layers to stack in the '
                                   'encoder/decoder models')
    parser_train.add_argument('-l', '--nonlinearity', choices=['tanh', 'relu'],
                              default='tanh', help='activation function to '
                                                   'use. For RNN model only')
    parser_train.add_argument('-b', '--bias', action='store_true',
                              help='whether or not to use biases in '
                                   'encoder/decoder models')
    parser_train.add_argument('-o', '--dropout', type=float, default=0,
                              help='probability in Dropout layer')
    parser_train.add_argument('-i', '--bidirectional', action='store_true',
                              help='if set, will use a bidirectional model '
                                   'in both encoder and decoder models')
    parser_train.add_argument('-r', '--learning-rate', type=float,
                              default=0.01, help='learning rate')
    parser_train.add_argument('-e', '--epochs', type=int, default=1,
                              help='number of epochs')
    parser_train.add_argument('-p', '--print-every', type=int, default=1000,
                              help='how often to print out loss information')
    parser_train.add_argument('-u', '--use-teacher-forcing',
                              action='store_true',
                              help='if set, will use teacher forcing')
    parser_train.add_argument('-f', '--teacher-forcing-ratio', type=float,
                              default=0.5, help='teacher forcing ratio')
    # parser_decode = subparsers.add_parser(
    #     'decode', formatter_class=argparse.RawTextHelpFormatter,
    #     help='decode input string: convert to simplified ortography')
    # parser_decode.set_defaults(func=decode)
    args = parser.parse_args()
    args.func(args)
