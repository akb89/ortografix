"""Train on a biased set of word-pairs. Evaluate on a unique set of wp.

Test set not in train. Decoder WITHOUT attention.

Learn to predict English -> Soundspel.
"""

import random
import statistics as stats

import ortografix
from ortografix import Encoder, Dataset, Decoder

if __name__ == '__main__':
    NUM_XP = 5
    EPOCHS = 10
    HIDDEN_SIZE = 128
    MODEL_TYPE = 'gru'
    NUM_LAYERS = 1
    BIDIRECTIONAL = False
    LEARNING_RATE = 0.01
    TEACHER_FORCING_RATIO = 0.5
    DATA_FILEPATH = '/home/kabbach/ortografix/data/experts.students.sync.all.as.wordpairs.txt'
    # DATA_FILEPATH = '/Users/akb/Github/ortografix/data/soundspel/experts.students.sync.all.as.wordpairs.txt'
    SHUFFLE = True
    ITEMIZE = False
    MAX_SEQ_LEN = 0
    REVERSE = False
    NON_LINEARITY = 'relu'
    BIAS = True
    DROPOUT = 0
    PRINT_EVERY = 1000
    MIN_COUNT = 2
    nsims = []
    total_dists = []
    pairs = []
    with open(DATA_FILEPATH, 'r', encoding='utf-8') as input_str:
        for line in input_str:
            line = line.strip()
            tokens = line.split('\t')
            pairs.append((tokens[0], tokens[1]))
    unique_pairs = set(pairs)
    num_test_items = round(.2 * len(unique_pairs))
    print('Input data contain {} unique pairs. Keeping {} for train and {} for test'
          .format(len(unique_pairs), len(unique_pairs)-num_test_items, num_test_items))
    for xp in range(NUM_XP):
        test_pairs = random.sample(unique_pairs, num_test_items)
        train_pairs = [x for x in pairs if x not in test_pairs]
        print('Biased training pairs = {}'.format(len(train_pairs)))
        dataset = Dataset(train_pairs, SHUFFLE, MAX_SEQ_LEN, REVERSE, MIN_COUNT)
        test_indexed_pairs = ortografix.index_pairs(
            test_pairs, dataset.left_vocab.char2idx, dataset.right_vocab.char2idx)
        encoder = Encoder(model_type=MODEL_TYPE,
                          input_size=dataset.left_vocab.size,
                          hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS,
                          nonlinearity=NON_LINEARITY,
                          bias=BIAS, dropout=DROPOUT,
                          bidirectional=BIDIRECTIONAL).to(ortografix.DEVICE)
        decoder = Decoder(model_type=MODEL_TYPE,
                          hidden_size=HIDDEN_SIZE,
                          output_size=dataset.right_vocab.size,
                          num_layers=NUM_LAYERS,
                          nonlinearity=NON_LINEARITY,
                          bias=BIAS, dropout=DROPOUT).to(ortografix.DEVICE)
        ortografix.train(encoder, decoder, dataset.indexed_pairs,
                         dataset.max_seq_len, EPOCHS, LEARNING_RATE,
                         PRINT_EVERY, TEACHER_FORCING_RATIO)
        total_dist, _, nsim, _ = ortografix.evaluate(
            test_indexed_pairs, ITEMIZE, encoder, decoder,
            dataset.right_vocab.idx2char, dataset.max_seq_len)
        total_dists.append(total_dist)
        nsims.append(nsim)
    print('avg dist = {}'.format(stats.mean(total_dists)))
    if len(total_dists) > 1:
        print('std dist = {}'.format(stats.stdev(total_dists)))
    print('avg nsim = {}'.format(stats.mean(nsims)))
    if len(nsims) > 1:
        print('std nsim = {}'.format(stats.stdev(nsims)))
