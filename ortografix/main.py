"""Welcome to ortografix.

This is the entry point of the application.
"""
import os

import argparse
import random
import time
import statistics
import logging
import logging.config

import textdistance
import torch
from torch import optim

import ortografix.utils.config as cutils
import ortografix.utils.constants as const
import ortografix.utils.time as tutils
import ortografix.utils.processing as putils

from ortografix.model.attention import Attention
from ortografix.model.encoder import Encoder
from ortografix.model.decoder import Decoder
from ortografix.model.dataset import Dataset, Vocab


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('train', 'evaluate')


def save_dataset_and_models(output_dirpath, dataset, encoder, decoder, loss,
                            learning_rate, with_attention):
    """Dump pytorch model and Dataset parameters."""
    logger.info('Saving dataset and models...')
    dataset.save_params(output_dirpath)
    model_params_filepath = os.path.join(output_dirpath, 'model.params')
    with open(model_params_filepath, 'w', encoding='utf-8') as m_params_str:
        print('cuda\t{}'.format(torch.cuda.is_available()), file=m_params_str)
    torch.save({'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder': {
                    'model_type': encoder.model_type,
                    'input_size': encoder.input_size,
                    'hidden_size': encoder.hidden_size,
                    'num_layers': encoder.num_layers,
                    'nonlinearity': encoder.nonlinearity,
                    'bias': encoder.bias,
                    'dropout': encoder.dropout,
                    'bidirectional': encoder.bidirectional
                },
                'decoder': {
                    'model_type': decoder.model_type,
                    'output_size': decoder.output_size,
                    'hidden_size': decoder.hidden_size,
                    'num_layers': decoder.num_layers,
                    'nonlinearity': decoder.nonlinearity,
                    'bias': decoder.bias,
                    'dropout': decoder.dropout,
                    'bidirectional': decoder.bidirectional
                },
                'with_attention': with_attention,
                'loss': loss,
                'learning_rate': learning_rate},
               os.path.join(output_dirpath, 'checkpoint.tar'))


# pylint: disable=R0914,E1102
def _train_single_batch(source_tensor, target_tensor, encoder, decoder,
                        with_attention, encoder_optimizer, decoder_optimizer,
                        max_seq_len, criterion, use_teacher_forcing,
                        teacher_forcing_ratio):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    # add 2 to max_seq_len to include SOS and EOS
    encoder_outputs = torch.zeros(max_seq_len+2, encoder.hidden_size,
                                  device=const.DEVICE)
    loss = 0
    for eidx in range(input_length):
        encoder_output, encoder_hidden = encoder(source_tensor[eidx],
                                                 encoder_hidden)
        encoder_outputs[eidx] = encoder_output[0, 0]
    decoder_input = torch.tensor([[const.SOS_IDX]], device=const.DEVICE)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for didx in range(target_length):
            if with_attention:
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[didx])
            decoder_input = target_tensor[didx]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for didx in range(target_length):
            if with_attention:
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            # detach from history as input
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[didx])
            if decoder_input.item() == const.EOS_IDX:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def _train(encoder, decoder, dataset, with_attention, num_epochs,
           learning_rate, print_every, use_teacher_forcing,
           teacher_forcing_ratio, output_dirpath):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    num_iter = 0
    num_total_iters = len(dataset.indexes) * num_epochs
    try:
        logger.info('Starting training...')
        for epoch in range(1, num_epochs+1):
            for source_indexes, target_indexes in dataset.indexes:
                source_tensor = torch.tensor(source_indexes, dtype=torch.long,
                                             device=const.DEVICE).view(-1, 1)
                target_tensor = torch.tensor(target_indexes, dtype=torch.long,
                                             device=const.DEVICE).view(-1, 1)
                num_iter += 1
                loss = _train_single_batch(
                    source_tensor, target_tensor, encoder, decoder,
                    with_attention, encoder_optimizer, decoder_optimizer,
                    dataset.max_seq_len, criterion, use_teacher_forcing,
                    teacher_forcing_ratio)
                print_loss_total += loss
                if num_iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    time_info = tutils.time_since(start,
                                                  num_iter / num_total_iters)
                    logger.info('Epoch {}/{} {} ({} {}%) {}'.format(
                        epoch, num_epochs, time_info, num_iter,
                        round(num_iter / num_total_iters * 100),
                        round(print_loss_avg, 4)))
        save_dataset_and_models(output_dirpath, dataset, encoder, decoder,
                                loss, learning_rate, with_attention)
    except KeyboardInterrupt:
        logger.info('Training interrupted')
        save_dataset_and_models(output_dirpath, dataset, encoder, decoder,
                                loss, learning_rate, with_attention)


def train(args):
    """Train the model."""
    logger.info('Training model from {}'.format(args.data))
    if not os.path.exists(args.output_dirpath):
        logger.info('Creating output directory to save files to: {}'
                    .format(args.output_dirpath))
        os.makedirs(args.output_dirpath, exist_ok=True)
    if torch.cuda.is_available():
        logger.info('Training on GPU')
    else:
        logger.info('No GPU available. Training on CPU')
    if not args.max_seq_len:
        max_seq_len = putils.get_max_seq_len(args.data, args.character_based)
        logger.info('max_seq_len not specified in args. Automatically setting '
                    'to longest source sequence length = {}'
                    .format(max_seq_len))
    else:
        logger.info('Manually setting max_seq_len to {}'
                    .format(args.max_seq_len))
        max_seq_len = args.max_seq_len
    dataset = Dataset(args.data, args.character_based, args.shuffle,
                      max_seq_len, args.reverse)
    encoder = Encoder(model_type=args.model_type,
                      input_size=dataset.source_vocab.size,
                      hidden_size=args.hidden_size,
                      num_layers=args.num_layers,
                      nonlinearity=args.nonlinearity,
                      bias=args.bias, dropout=args.dropout,
                      bidirectional=args.bidirectional).to(const.DEVICE)
    if args.with_attention:
        decoder = Attention(model_type=args.model_type,
                            hidden_size=args.hidden_size,
                            output_size=dataset.target_vocab.size,
                            max_seq_len=dataset.max_seq_len,
                            num_layers=args.num_layers,
                            nonlinearity=args.nonlinearity,
                            bias=args.bias, dropout=args.dropout,
                            bidirectional=args.bidirectional).to(const.DEVICE)
    else:
        decoder = Decoder(model_type=args.model_type,
                          hidden_size=args.hidden_size,
                          output_size=dataset.target_vocab.size,
                          num_layers=args.num_layers,
                          nonlinearity=args.nonlinearity,
                          bias=args.bias, dropout=args.dropout,
                          bidirectional=args.bidirectional).to(const.DEVICE)
    return _train(encoder, decoder, dataset, args.with_attention, args.epochs,
                  args.learning_rate, args.print_every,
                  args.use_teacher_forcing, args.teacher_forcing_ratio,
                  args.output_dirpath)


def _decode(source_indexes, encoder, decoder, with_attention, max_seq_len):
    with torch.no_grad():
        input_tensor = torch.tensor(source_indexes, dtype=torch.long,
                                    device=const.DEVICE).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()
        # add 2 to max_seq_len to include SOS and EOS
        encoder_outputs = torch.zeros(max_seq_len+2, encoder.hidden_size,
                                      device=const.DEVICE)
        for eidx in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[eidx], encoder_hidden)
            encoder_outputs[eidx] += encoder_output[0, 0]
        decoder_input = torch.tensor([[const.SOS_IDX]], device=const.DEVICE)
        decoder_hidden = encoder_hidden
        decoded_indexes = []
        # add 2 to max_seq_len to include SOS and EOS
        decoder_attentions = torch.zeros(max_seq_len+2, max_seq_len+2)
        for didx in range(max_seq_len+2):
            if with_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[didx] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input,
                                                         decoder_hidden)
            _, topi = decoder_output.data.topk(1)
            if topi.item() == const.EOS_IDX:
                decoded_indexes.append(const.EOS_IDX)
                break
            decoded_indexes.append(topi.item())
            decoder_input = topi.squeeze().detach()
        return decoded_indexes, decoder_attentions[:didx + 1]


def _evaluate(indexes, encoder, decoder, target_vocab, checkpoint,
              dataset_params):
    avg_dist = []
    avg_norm_dist = []
    avg_norm_sim = []
    for seq in indexes:
        pred_idx, _ = _decode(seq[0], encoder, decoder,
                              checkpoint['with_attention'],
                              dataset_params['max_seq_len'])
        gold = ''.join(
            [target_vocab.idx2item[idx] for idx in seq[1] if idx not in
             [const.SOS_IDX, const.EOS_IDX]])
        prediction = ''.join(
            [target_vocab.idx2item[idx] for idx in pred_idx if idx not in
             [const.SOS_IDX, const.EOS_IDX]])
        avg_dist.append(textdistance.levenshtein.distance(gold,
                                                          prediction))
        avg_norm_dist.append(textdistance.levenshtein.normalized_distance(
            gold, prediction))
        avg_norm_sim.append(textdistance.levenshtein.normalized_similarity(
            gold, prediction))
    logger.info('Printing Levenshtein edit distance info:')
    logger.info('   total sum dist = {}'.format(sum(avg_dist)))
    logger.info('   avg dist = {}'.format(statistics.mean(avg_dist)))
    # logger.info('   avg sim = {}'.format(statistics.mean(avg_sim)))
    logger.info('   avg normalized dist = {}'
                .format(statistics.mean(avg_norm_dist)))
    logger.info('   avg normalized sim = {}'
                .format(statistics.mean(avg_norm_sim)))
    return avg_dist, avg_norm_dist, avg_norm_sim


def evaluate(args):
    """Evaluate a given model on a test set."""
    dataset_param_filepath = os.path.join(args.model, 'dataset.params')
    dataset_params = putils.load_params(dataset_param_filepath)
    source_vocab_filepath = os.path.join(args.model, 'source.vocab')
    source_vocab = Vocab(vocab_filepath=source_vocab_filepath)
    target_vocab_filepath = os.path.join(args.model, 'target.vocab')
    target_vocab = Vocab(vocab_filepath=target_vocab_filepath)
    model_params_filepath = os.path.join(args.model, 'model.params')
    model_params = putils.load_params(model_params_filepath)
    checkpoint_filepath = os.path.join(args.model, 'checkpoint.tar')
    if not torch.cuda.is_available() and model_params['cuda']:
        logger.info('Loading a GPU-trained model on CPU')
        checkpoint = torch.load(checkpoint_filepath,
                                map_location=const.DEVICE)
    elif torch.cuda.is_available() and model_params['cuda']:
        logger.info('Loading a GPU-trained model on GPU')
        checkpoint = torch.load(checkpoint_filepath)
    elif torch.cuda.is_available() and not model_params['cuda']:
        logger.info('Loading a CPU-trained model on GPU')
        checkpoint = torch.load(checkpoint_filepath,
                                map_location='cuda:0')
    else:
        logger.info('Loading a CPU-trained model on CPU')
        checkpoint = torch.load(checkpoint_filepath)
    encoder = Encoder(model_type=checkpoint['encoder']['model_type'],
                      input_size=checkpoint['encoder']['input_size'],
                      hidden_size=checkpoint['encoder']['hidden_size'],
                      num_layers=checkpoint['encoder']['num_layers'],
                      nonlinearity=checkpoint['encoder']['nonlinearity'],
                      bias=checkpoint['encoder']['bias'],
                      dropout=checkpoint['encoder']['dropout'],
                      bidirectional=checkpoint['encoder']['bidirectional'])
    if checkpoint['with_attention']:
        decoder = Attention(model_type=checkpoint['decoder']['model_type'],
                            hidden_size=checkpoint['decoder']['hidden_size'],
                            output_size=checkpoint['decoder']['output_size'],
                            max_seq_len=dataset_params['max_seq_len'],
                            num_layers=checkpoint['decoder']['num_layers'],
                            nonlinearity=checkpoint['decoder']['nonlinearity'],
                            bias=checkpoint['decoder']['bias'],
                            dropout=checkpoint['decoder']['dropout'],
                            bidirectional=checkpoint['decoder']['bidirectional'])
    else:
        decoder = Decoder(model_type=checkpoint['decoder']['model_type'],
                          hidden_size=checkpoint['decoder']['hidden_size'],
                          output_size=checkpoint['decoder']['output_size'],
                          num_layers=checkpoint['decoder']['num_layers'],
                          nonlinearity=checkpoint['decoder']['nonlinearity'],
                          bias=checkpoint['decoder']['bias'],
                          dropout=checkpoint['decoder']['dropout'],
                          bidirectional=checkpoint['decoder']['bidirectional'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if torch.cuda.is_available():
        encoder.to(const.DEVICE)
        decoder.to(const.DEVICE)
    encoder.eval()
    decoder.eval()
    indexes = putils.index_dataset(
        args.data, source_vocab.item2idx, target_vocab.item2idx,
        dataset_params['is_character_based'], dataset_params['max_seq_len'],
        dataset_params['is_reversed'])
    if args.random > 0:
        random.shuffle(indexes)
        for seq_num in range(args.random):
            seq = indexes[seq_num]
            print('-'*80)
            print('>', ' '.join([source_vocab.idx2item[idx]
                                 for idx in seq[0]]))
            print('=', ' '.join([target_vocab.idx2item[idx]
                                 for idx in seq[1]]))
            # TODO: add support for OOV
            predicted_idx, _ = _decode(seq[0], encoder, decoder,
                                       checkpoint['with_attention'],
                                       dataset_params['max_seq_len'])
            print('<', ' '.join([target_vocab.idx2item[idx]
                                 for idx in predicted_idx]))
    else:
        _evaluate(indexes, encoder, decoder, target_vocab, checkpoint,
                  dataset_params)


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
    parser_train.add_argument('-m', '--max-seq-len', type=int, default=0,
                              help='maximum sequence length to retain. If not '
                                   'set manually, will be set to the length '
                                   'of the longest sequence in the dataset')
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
    parser_train.add_argument('-a', '--output-dirpath', required=True,
                              help='absolute dirpath where to save models')
    parser_train.add_argument('-w', '--with-attention', action='store_true',
                              help='if set, will use attention-based decoding')
    parser_train.add_argument('-v', '--reverse', action='store_true',
                              help='if set, will reverse dataset pairs')
    parser_evaluate = subparsers.add_parser(
        'evaluate', formatter_class=argparse.RawTextHelpFormatter,
        help='evaluate model on input test set')
    parser_evaluate.set_defaults(func=evaluate)
    parser_evaluate.add_argument('-m', '--model', required=True,
                                 help='absolute path to model directory')
    parser_evaluate.add_argument('-d', '--data', required=True,
                                 help='absolute path to test set')
    parser_evaluate.add_argument('-v', '--reverse', action='store_true',
                                 help='if set, will reverse dataset pairs')
    parser_evaluate.add_argument('-r', '--random', type=int, default=0,
                                 help='if > 0, will test on n sequences '
                                      'randomly selected from test set')
    args = parser.parse_args()
    args.func(args)
