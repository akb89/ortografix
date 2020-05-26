"""Welcome to ortografix.

This is the entry point of the application.
"""
import os

import argparse
import random
import time
import itertools
import logging
import logging.config

import statistics as stats

import torch
from torch import optim
import textdistance as dist

import ortografix.utils.config as cutils
import ortografix.utils.constants as const
import ortografix.utils.time as tutils
import ortografix.utils.processing as putils

from ortografix.model.attention import Attention
from ortografix.model.encoder import Encoder
from ortografix.model.decoder import Decoder
from ortografix.model.dataset import Dataset, Vocab
from ortografix.model.transformer import TEncoder, TDecoder


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('train', 'evaluate')


def save_dataset_and_models(output_dirpath, dataset, encoder, decoder, loss,
                            learning_rate):
    """Dump pytorch model and Dataset parameters."""
    logger.info('Saving dataset and models...')
    dataset.save_params(output_dirpath)
    model_params_filepath = os.path.join(output_dirpath, 'model.params')
    with open(model_params_filepath, 'w', encoding='utf-8') as m_params_str:
        print('cuda\t{}'.format(torch.cuda.is_available()), file=m_params_str)
    if encoder.model_type == 'transformer':
        torch.save({'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder': {
                        'model_type': encoder.model_type,
                        'input_size': encoder.input_size,
                        'hidden_size': encoder.hidden_size,
                        'num_layers': encoder.num_layers,
                        'dropout': encoder.dropout,
                        'num_attention_heads': encoder.num_attention_heads
                    },
                    'decoder': {
                        'model_type': decoder.model_type,
                        'output_size': decoder.output_size,
                        'hidden_size': decoder.hidden_size,
                        'num_layers': decoder.num_layers,
                        'dropout': decoder.dropout,
                        'num_attention_heads': encoder.num_attention_heads
                    },
                    'loss': loss,
                    'learning_rate': learning_rate},
                   os.path.join(output_dirpath, 'checkpoint.tar'))
    else:
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
                        'with_attention': decoder.with_attention
                    },
                    'loss': loss,
                    'learning_rate': learning_rate},
                   os.path.join(output_dirpath, 'checkpoint.tar'))


# pylint: disable=R0914,E1102
def _train_single_batch(source_tensor, target_tensor, encoder, decoder,
                        encoder_optimizer, decoder_optimizer,
                        max_seq_len, criterion, teacher_forcing_ratio,
                        last_decoder_pred_idx):
    if encoder.model_type != 'transformer':
        encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    if encoder.model_type == 'transformer' or not encoder.bidirectional:
        encoder_outputs = torch.zeros(max_seq_len, encoder.hidden_size,
                                      device=const.DEVICE)
    else:
        encoder_outputs = torch.zeros(max_seq_len, encoder.hidden_size*2,
                                      device=const.DEVICE)

    loss = 0
    for eidx in range(input_length):
        if encoder.model_type == 'transformer':
            encoder_output = encoder(source_tensor[eidx])
        else:
            encoder_output, encoder_hidden = encoder(source_tensor[eidx],
                                                     encoder_hidden)
        encoder_outputs[eidx] = encoder_output[0, 0]
    decoder_input = torch.tensor([[last_decoder_pred_idx]],
                                 device=const.DEVICE)
    if decoder.model_type != 'transformer':
        if encoder.bidirectional:
            # here we use summing rather than conctenation to keep same dim in decoder
            # https://discuss.pytorch.org/t/about-bidirectional-gru-with-seq2seq-example-and-some-modifications/15588/5
            encoder_outputs = encoder_outputs[:, :encoder.hidden_size] + encoder_outputs[:, encoder.hidden_size:]
            decoder_hidden = encoder_hidden[-decoder.num_layers:]
        else:
            decoder_hidden = encoder_hidden
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for didx in range(target_length):
            if decoder.model_type == 'transformer':
                decoder_output = decoder(decoder_input, encoder_outputs)
            elif decoder.with_attention:
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
            if decoder.model_type == 'transformer':
                decoder_output = decoder(decoder_input, encoder_outputs)
            elif decoder.with_attention:
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
    # if encoder.model_type == 'transformer':
    #     torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
    #     torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length, decoder_input.item()


def _train(encoder, decoder, indexed_pairs, max_seq_len, num_epochs,
           learning_rate, print_every, teacher_forcing_ratio):
    criterion = torch.nn.NLLLoss()
    # if encoder.model_type == 'transformer':
    #     encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=5.0)
    #     encoder_scheduler = torch.optim.lr_scheduler.StepLR(
    #         encoder_optimizer, 1.0, gamma=0.95)
    #     decoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=5.0)
    #     decoder_scheduler = torch.optim.lr_scheduler.StepLR(
    #         decoder_optimizer, 1.0, gamma=0.95)
    # else:
    #     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    num_iter = 0
    num_total_iters = len(indexed_pairs) * num_epochs
    for epoch in range(1, num_epochs+1):
        for source_indexes, target_indexes in indexed_pairs:
            last_decoder_pred_idx = const.SOS_IDX
            for idx in range(0, min(len(source_indexes),
                                    len(target_indexes)), max_seq_len):
                if idx + max_seq_len > len(source_indexes):
                    source = source_indexes[idx:]
                else:
                    source = source_indexes[idx: idx+max_seq_len]
                if idx + max_seq_len > len(target_indexes):
                    target = target_indexes[idx:]
                else:
                    target = target_indexes[idx: idx+max_seq_len]
                source_tensor = torch.tensor(source, dtype=torch.long,
                                             device=const.DEVICE).view(-1, 1)
                target_tensor = torch.tensor(target, dtype=torch.long,
                                             device=const.DEVICE).view(-1, 1)
                num_iter += 1
                loss, last_decoder_pred_idx = _train_single_batch(
                    source_tensor, target_tensor, encoder, decoder,
                    encoder_optimizer, decoder_optimizer,
                    max_seq_len, criterion, teacher_forcing_ratio,
                    last_decoder_pred_idx)
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
        # if encoder.model_type == 'transformer':
        #     encoder_scheduler.step()
        #     decoder_scheduler.step()
    return encoder, decoder, loss


def train(args):
    """Train the model."""
    logger.info('Training model from {}'.format(args.data))
    if not args.output_dirpath:
        logger.warning('You haven\'t specified --output-dirpath. Models will '
                       'not be saved!')
    elif not os.path.exists(args.output_dirpath):
        logger.info('Creating output directory to save files to: {}'
                    .format(args.output_dirpath))
        os.makedirs(args.output_dirpath, exist_ok=True)
    if torch.cuda.is_available():
        logger.info('Training on GPU')
    else:
        logger.info('No GPU available. Training on CPU')
    seq_pairs = putils.convert_to_seq_pairs(args.data)
    dataset = Dataset(seq_pairs, args.shuffle, args.max_seq_len,
                      args.min_count, args.reverse)
    if args.reverse:
        indexed_pairs = [(y, x) for x, y in dataset.indexed_pairs]
        enc_input_size = dataset.right_vocab.size
        dec_input_size = dataset.left_vocab.size
    else:
        indexed_pairs = dataset.indexed_pairs
        enc_input_size = dataset.left_vocab.size
        dec_input_size = dataset.right_vocab.size
    if args.model_type == 'transformer':
        encoder = TEncoder(input_size=enc_input_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           num_attention_heads=args.num_attention_heads).to(const.DEVICE)
    else:
        encoder = Encoder(model_type=args.model_type,
                          input_size=enc_input_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          nonlinearity=args.nonlinearity,
                          bias=args.bias, dropout=args.dropout,
                          bidirectional=args.bidirectional).to(const.DEVICE)
    if args.model_type == 'transformer':
        decoder = TDecoder(hidden_size=args.hidden_size,
                           output_size=dec_input_size,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           num_attention_heads=args.num_attention_heads).to(const.DEVICE)
    elif args.with_attention:
        decoder = Attention(hidden_size=args.hidden_size,
                            output_size=dec_input_size,
                            max_seq_len=dataset.max_seq_len,
                            num_layers=args.num_layers,
                            nonlinearity=args.nonlinearity,
                            bias=args.bias,
                            dropout=args.dropout).to(const.DEVICE)
    else:
        decoder = Decoder(model_type=args.model_type,
                          hidden_size=args.hidden_size,
                          output_size=dec_input_size,
                          num_layers=args.num_layers,
                          nonlinearity=args.nonlinearity,
                          bias=args.bias,
                          dropout=args.dropout).to(const.DEVICE)
    encoder, decoder, loss = _train(
        encoder, decoder, indexed_pairs, dataset.max_seq_len,
        args.epochs, args.learning_rate, args.print_every,
        args.teacher_forcing_ratio)
    if args.output_dirpath:
        save_dataset_and_models(args.output_dirpath, dataset, encoder, decoder,
                                loss, args.learning_rate)


def _decode(source_indexes, encoder, decoder, max_seq_len):
    with torch.no_grad():
        if encoder.model_type != 'transformer':
            encoder_hidden = encoder.init_hidden()
        last_decoder_pred_idx = const.SOS_IDX
        decoded_indexes = []
        for idx in range(0, len(source_indexes), max_seq_len):
            if idx + max_seq_len > len(source_indexes):
                source = source_indexes[idx:]
            else:
                source = source_indexes[idx: idx+max_seq_len]
            input_tensor = torch.tensor(source, dtype=torch.long,
                                        device=const.DEVICE).view(-1, 1)
            input_length = input_tensor.size()[0]
            if encoder.model_type == 'transformer' or not encoder.bidirectional:
                encoder_outputs = torch.zeros(max_seq_len, encoder.hidden_size,
                                              device=const.DEVICE)
            else:
                encoder_outputs = torch.zeros(max_seq_len,
                                              encoder.hidden_size*2,
                                              device=const.DEVICE)
            for eidx in range(input_length):
                if encoder.model_type == 'transformer':
                    encoder_output = encoder(input_tensor[eidx])
                else:
                    encoder_output, encoder_hidden = encoder(input_tensor[eidx],
                                                             encoder_hidden)
                encoder_outputs[eidx] += encoder_output[0, 0]
            decoder_input = torch.tensor([[last_decoder_pred_idx]],
                                         device=const.DEVICE)
            if decoder.model_type != 'transformer':
                if encoder.bidirectional:
                    # here we use summing rather than conctenation to keep same dim in decoder
                    # https://discuss.pytorch.org/t/about-bidirectional-gru-with-seq2seq-example-and-some-modifications/15588/5
                    encoder_outputs = encoder_outputs[:, :encoder.hidden_size] + encoder_outputs[:, encoder.hidden_size:]
                    decoder_hidden = encoder_hidden[-decoder.num_layers:]
                else:
                    decoder_hidden = encoder_hidden
            decoder_attentions = torch.zeros(max_seq_len, max_seq_len)
            for didx in range(max_seq_len):
                if decoder.model_type == 'transformer':
                    decoder_output = decoder(decoder_input, encoder_outputs)
                elif decoder.with_attention:
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
            last_decoder_pred_idx = decoder_input.item()
        return decoded_indexes, decoder_attentions[:didx + 1]


def decode(sequence, itemize, encoder, decoder, max_seq_len):
    predicted_idxx = []
    if itemize:
        key = lambda sep: sep == const.SEP_IDX
        seqx = [list(group) for is_key, group in
                itertools.groupby(sequence, key) if not is_key]
        for chars in seqx:
            if chars[0] != const.SOS_IDX:
                chars.insert(0, const.SOS_IDX)
            if chars[-1] != const.EOS_IDX:
                chars.append(const.EOS_IDX)
            predicted_idx, _ = _decode(chars, encoder, decoder, max_seq_len)
            predicted_idxx.extend(predicted_idx)
            predicted_idxx.append(const.SEP_IDX)
        predicted_idxx.pop()

    else:
        predicted_idxx, _ = _decode(sequence, encoder, decoder, max_seq_len)
    return predicted_idxx


def _evaluate(indexed_pairs, itemize, encoder, decoder, idx2char, max_seq_len):
    avg_dist = []
    avg_dl = []
    # avg_norm_dist = []
    nsim = []
    dl_nsim = []
    for seq in indexed_pairs:
        pred_idx = decode(seq[0], itemize, encoder, decoder, max_seq_len)
        gold = ' '.join(''.join(
            [idx2char[idx] for idx in seq[1] if idx not in
             [const.SOS_IDX, const.EOS_IDX]]).split(const.SEP))
        prediction = ' '.join(''.join(
            [idx2char[idx] for idx in pred_idx if idx not in
             [const.SOS_IDX, const.EOS_IDX]]).split(const.SEP))
        avg_dist.append(dist.levenshtein.distance(gold, prediction))
        avg_dl.append(dist.damerau_levenshtein.distance(gold, prediction))
        # avg_norm_dist.append(dist.levenshtein.normalized_distance(gold,
        #                                                           prediction))
        nsim.append(dist.levenshtein.normalized_similarity(gold, prediction))
        dl_nsim.append(dist.damerau_levenshtein.normalized_similarity(
            gold, prediction))
    logger.info('Printing edit distance info:')
    logger.info('   total sum dist = {}'.format(sum(avg_dist)))
    logger.info('   total sum DL = {}'.format(sum(avg_dl)))
    logger.info('   avg dist = {}'.format(stats.mean(avg_dist)))
    logger.info('   avg DL = {}'.format(stats.mean(avg_dl)))
    # logger.info('   avg normalized dist = {}'
    #             .format(stats.mean(avg_norm_dist)))
    logger.info('   avg normalized sim = {}'.format(stats.mean(nsim)))
    logger.info('   avg normalized DL sim = {}'.format(stats.mean(dl_nsim)))
    return stats.mean(avg_dist), stats.mean(nsim), stats.mean(dl_nsim)


def evaluate(args):
    """Evaluate a given model on a test set."""
    dataset_param_filepath = os.path.join(args.model, 'dataset.params')
    dataset_params = putils.load_params(dataset_param_filepath)
    left_vocab_filepath = os.path.join(args.model, 'left.vocab')
    left_vocab = Vocab(vocab_filepath=left_vocab_filepath)
    right_vocab_filepath = os.path.join(args.model, 'right.vocab')
    right_vocab = Vocab(vocab_filepath=right_vocab_filepath)
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
    if checkpoint['encoder']['model_type'] == 'transformer':
        encoder = TEncoder(input_size=checkpoint['encoder']['input_size'],
                           hidden_size=checkpoint['encoder']['hidden_size'],
                           num_layers=checkpoint['encoder']['num_layers'],
                           dropout=checkpoint['encoder']['dropout'],
                           num_attention_heads=checkpoint['encoder']['num_attention_heads'])
    else:
        encoder = Encoder(model_type=checkpoint['encoder']['model_type'],
                          input_size=checkpoint['encoder']['input_size'],
                          hidden_size=checkpoint['encoder']['hidden_size'],
                          num_layers=checkpoint['encoder']['num_layers'],
                          nonlinearity=checkpoint['encoder']['nonlinearity'],
                          bias=checkpoint['encoder']['bias'],
                          dropout=checkpoint['encoder']['dropout'],
                          bidirectional=checkpoint['encoder']['bidirectional'])
    if checkpoint['decoder']['model_type'] == 'transformer':
        decoder = TDecoder(hidden_size=checkpoint['decoder']['hidden_size'],
                           output_size=checkpoint['decoder']['output_size'],
                           num_layers=checkpoint['decoder']['num_layers'],
                           dropout=checkpoint['decoder']['dropout'],
                           num_attention_heads=checkpoint['decoder']['num_attention_heads'])
    elif checkpoint['with_attention']:
        decoder = Attention(hidden_size=checkpoint['decoder']['hidden_size'],
                            output_size=checkpoint['decoder']['output_size'],
                            max_seq_len=dataset_params['max_seq_len'],
                            num_layers=checkpoint['decoder']['num_layers'],
                            nonlinearity=checkpoint['decoder']['nonlinearity'],
                            bias=checkpoint['decoder']['bias'],
                            dropout=checkpoint['decoder']['dropout'])
    else:
        decoder = Decoder(model_type=checkpoint['decoder']['model_type'],
                          hidden_size=checkpoint['decoder']['hidden_size'],
                          output_size=checkpoint['decoder']['output_size'],
                          num_layers=checkpoint['decoder']['num_layers'],
                          nonlinearity=checkpoint['decoder']['nonlinearity'],
                          bias=checkpoint['decoder']['bias'],
                          dropout=checkpoint['decoder']['dropout'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if torch.cuda.is_available():
        encoder.to(const.DEVICE)
        decoder.to(const.DEVICE)
    encoder.eval()
    decoder.eval()
    pairs = putils.convert_to_seq_pairs(args.data)
    indexed_pairs = putils.index_pairs(pairs, left_vocab.char2idx,
                                       right_vocab.char2idx)
    if dataset_params['reverse']:
        raise Exception('Unsupported reverse option')
    if args.random > 0:
        random.shuffle(indexed_pairs)
        for seq_num in range(args.random):
            seq = indexed_pairs[seq_num]
            print('-'*80)
            input_str = ' '.join(
                ''.join([left_vocab.idx2char[idx] for idx in seq[0] if idx
                         not in [const.SOS_IDX, const.EOS_IDX]])
                .split(const.SEP))
            gold_str = ' '.join(
                ''.join([right_vocab.idx2char[idx] for idx in seq[1] if idx
                         not in [const.SOS_IDX, const.EOS_IDX]])
                .split(const.SEP))
            predicted_idxx = decode(seq[0], args.itemize, encoder, decoder,
                                    dataset_params['max_seq_len'])
            pred_str = ' '.join(
                ''.join([right_vocab.idx2char[idx] for idx in predicted_idxx
                         if idx not in [const.SOS_IDX, const.EOS_IDX]])
                .split(const.SEP))
            print('>', input_str)
            print('=', gold_str)
            print('<', pred_str)
    else:
        _evaluate(indexed_pairs, args.itemize, encoder, decoder,
                  right_vocab.idx2char, dataset_params['max_seq_len'])


def convert(args):
    """Convert text file with aligned words and sent to word pairs."""
    if args.unique:
        output_filepath = '{}.as.unique.wordpairs.txt'.format(
            args.data.split('.txt')[0])
    else:
        output_filepath = '{}.as.wordpairs.txt'.format(
            args.data.split('.txt')[0])
    pairs = []
    logger.info('Saving output to {}'.format(output_filepath))
    with open(args.data, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            seq = line.split('\t')
            if len(seq[0].split()) == 1:
                pairs.append((seq[0], seq[1]))
            else:
                xtokens = seq[0].split()
                ytokens = seq[1].split()
                if len(xtokens) != len(ytokens):
                    raise Exception(
                        'Invalid input sequences: should contain the same '
                        'number of tokens: \n {} \n {}'.format(seq[0], seq[1]))
                for xtoken, ytoken in zip(xtokens, ytokens):
                    pairs.append((xtoken, ytoken))
    if args.unique:
        pairs = set(pairs)
    with open(output_filepath, 'w', encoding='utf-8') as output_str:
        for pair in sorted(pairs):
            print('{}\t{}'.format(pair[0], pair[1]), file=output_str)
    # with open(args.data, 'r', encoding='utf-8') as input_stream:
    #     with open(output_filepath, 'w', encoding='utf-8') as output_str:
    #         for line in input_stream:
    #             line = line.strip()
    #             if line:
    #                 seq = line.split('\t')
    #                 xtokens = seq[0].split()
    #                 ytokens = seq[1].split()
    #                 if len(xtokens) != len(ytokens):
    #                     raise Exception(
    #                         'Invalid input sequences: should contain the same '
    #                         'number of tokens: \n {} \n {}'.format(seq[0], seq[1]))
    #                 print('{}\t{}'.format(' '.join(xtokens), ' '.join(ytokens)),
    #                       file=output_str)


def main():
    """Launch ortografix."""
    parser = argparse.ArgumentParser(prog='ortografix')
    subparsers = parser.add_subparsers()
    parser_convert = subparsers.add_parser(
        'convert', formatter_class=argparse.RawTextHelpFormatter,
        help='convert text file with aligned words and sentences to list '
             'of unique word pairs')
    parser_convert.set_defaults(func=convert)
    parser_convert.add_argument('-d', '--data', required=True,
                                help='absolute path to input file to convert')
    parser_convert.add_argument('-u', '--unique', action='store_true',
                                help='if set, will return unique pairs only')
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        help='train the seq2seq model')
    parser_train.set_defaults(func=train)
    parser_train.add_argument('-d', '--data', required=True,
                              help='absolute path to training data')
    parser_train.add_argument('-t', '--model-type',
                              choices=['rnn', 'gru', 'lstm', 'transformer'],
                              default='gru',
                              help='encoder/decoder model type')
    parser_train.add_argument('-s', '--shuffle', action='store_true',
                              help='if set, will shuffle the training data')
    parser_train.add_argument('-c', '--min-count', type=int, default=2,
                              help='min char count to be included in vocab')
    parser_train.add_argument('-q', '--max-seq-len', type=int, default=0,
                              help='maximum sequence length to retain. If not '
                                   'set manually, will be set to the length '
                                   'of the longest sequence in the dataset')
    parser_train.add_argument('-z', '--hidden-size', type=int, default=256,
                              help='size of the hidden layer')
    parser_train.add_argument('-n', '--num-layers', type=int, default=1,
                              help='number of layers to stack in the '
                                   'encoder/decoder models')
    parser_train.add_argument('-l', '--nonlinearity', choices=['tanh', 'relu'],
                              default='relu', help='activation function to '
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
    parser_train.add_argument('-f', '--teacher-forcing-ratio', type=float,
                              default=0.5, help='teacher forcing ratio')
    parser_train.add_argument('-a', '--output-dirpath',
                              help='absolute dirpath where to save models')
    parser_train.add_argument('-w', '--with-attention', action='store_true',
                              help='if set, will use attention-based decoding')
    parser_train.add_argument('-v', '--reverse', action='store_true',
                              help='if set, will reverse dataset pairs')
    parser_train.add_argument('-y', '--num-attention-heads', type=int,
                              default=2,
                              help='number of attention heads in the '
                                   'Transformer model')
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
    parser_evaluate.add_argument('-i', '--itemize', action='store_true',
                                 help='if set, will predict token-by-token'
                                      'in the sequence rather than the whole'
                                      'sequence at once')
    args = parser.parse_args()
    args.func(args)
