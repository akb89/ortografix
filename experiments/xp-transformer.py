"""XP."""

import random
import statistics as stats

import ortografix
from ortografix import Dataset, TEncoder, TDecoder

if __name__ == '__main__':
    NUM_XP = 5
    DATA_FILEPATH = '/home/debian/ortografix/data/experts.students.sync.all.as.wordpairs.txt'
    # DATA_FILEPATH = '/Users/akb/Github/ortografix/data/soundspel/experts.students.sync.all.as.wordpairs.txt'
    # OUTPUT_DIRPATH = '/Users/akb/Github/ortografix/models/xp001/'
    SHUFFLE = True
    ITEMIZE = False
    MAX_SEQ_LEN = 0
    REVERSE = False
    MODEL_TYPE = 'transformer'
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    NUM_ATTENTION_HEADS = 1
    DROPOUT = 0
    LEARNING_RATE = 0.01
    EPOCHS = 5
    TEACHER_FORCING_RATIO = 0.5
    PRINT_EVERY = 1000
    MIN_COUNT = 2
    nsims = []
    dl_nsims = []
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
        encoder = TEncoder(input_size=dataset.left_vocab.size,
                           hidden_size=HIDDEN_SIZE,
                           num_layers=NUM_LAYERS,
                           dropout=DROPOUT,
                           num_attention_heads=NUM_ATTENTION_HEADS).to(ortografix.DEVICE)
        decoder = TDecoder(hidden_size=HIDDEN_SIZE,
                           output_size=dataset.right_vocab.size,
                           num_layers=NUM_LAYERS,
                           dropout=DROPOUT,
                           num_attention_heads=NUM_ATTENTION_HEADS).to(ortografix.DEVICE)
        ortografix.train(encoder, decoder, dataset.indexed_pairs,
                         dataset.max_seq_len, EPOCHS, LEARNING_RATE,
                         PRINT_EVERY, TEACHER_FORCING_RATIO)
        _, nsim, dl_nsim = ortografix.evaluate(
            test_indexed_pairs, ITEMIZE, encoder, decoder,
            dataset.right_vocab.idx2char, dataset.max_seq_len)
        nsims.append(nsim)
        dl_nsims.append(dl_nsim)
    print('avg nsim = {}'.format(stats.mean(nsims)))
    if len(nsims) > 1:
        print('std nsim = {}'.format(stats.stdev(nsims)))
    print('avg DL nsim = {}'.format(stats.mean(dl_nsims)))
    if len(dl_nsims) > 1:
        print('std DL nsim = {}'.format(stats.stdev(dl_nsims)))
