"""bidirGRU with Attention on experts annotation."""

import random
import statistics as stats

import ortografix
from ortografix import Attention, Encoder, Dataset, Decoder

if __name__ == '__main__':
    NUM_XP = 10
    DATA_FILEPATH = '/home/kabbach/ortografix/data/experts.all.as.wordpairs.txt'
    # DATA_FILEPATH = '/Users/akb/Github/ortografix/data/soundspel/experts.all.as.wordpairs.txt'
    # OUTPUT_DIRPATH = '/Users/akb/Github/ortografix/models/xp001/'
    CHARACTER_BASED = True
    SHUFFLE = True
    MAX_SEQ_LEN = 0
    REVERSE = False
    MODEL_TYPE = 'gru'
    HIDDEN_SIZE = 256
    NUM_LAYERS = 1
    NON_LINEARITY = 'tanh'  # try relu later?
    BIAS = True
    DROPOUT = 0
    BIDIRECTIONAL = True
    LEARNING_RATE = 0.01
    EPOCHS = 10
    USE_TEACHER_FORCING = True
    TEACHER_FORCING_RATIO = 0.5
    WITH_ATTENTION = True
    PRINT_EVERY = 100
    MIN_COUNT = 2
    dists = []
    norm_dists = []
    norm_sims = []
    pairs = []
    with open(DATA_FILEPATH, 'r', encoding='utf-8') as input_str:
        for line in input_str:
            line = line.strip()
            tokens = line.split('\t')
            pairs.append((tokens[0], tokens[1]))
    num_test_items = round(.2 * len(pairs))
    print('Input data contain {} pairs. Keeping {} for train and {} for test'
          .format(len(pairs), len(pairs)-num_test_items, num_test_items))
    for xp in range(NUM_XP):
        test_pairs = random.sample(pairs, num_test_items)
        train_pairs = [x for x in pairs if x not in test_pairs]
        dataset = Dataset(train_pairs, CHARACTER_BASED, SHUFFLE, MAX_SEQ_LEN,
                          REVERSE, MIN_COUNT)
        test_indexes = ortografix.index_pairs(
            test_pairs, dataset.source_vocab.item2idx,
            dataset.target_vocab.item2idx, dataset.is_character_based,
            dataset.max_seq_len, dataset.is_reversed)
        encoder = Encoder(model_type=MODEL_TYPE,
                          input_size=dataset.source_vocab.size,
                          hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS,
                          nonlinearity=NON_LINEARITY,
                          bias=BIAS, dropout=DROPOUT,
                          bidirectional=BIDIRECTIONAL).to(ortografix.DEVICE)
        decoder = Attention(model_type=MODEL_TYPE,
                            hidden_size=HIDDEN_SIZE,
                            output_size=dataset.target_vocab.size,
                            max_seq_len=dataset.max_seq_len,
                            num_layers=NUM_LAYERS,
                            nonlinearity=NON_LINEARITY,
                            bias=BIAS, dropout=DROPOUT,
                            bidirectional=BIDIRECTIONAL).to(ortografix.DEVICE)
        ortografix.train(encoder, decoder, dataset.indexes,
                         dataset.max_seq_len, WITH_ATTENTION, EPOCHS,
                         LEARNING_RATE, PRINT_EVERY, USE_TEACHER_FORCING,
                         TEACHER_FORCING_RATIO)
        dist, norm_dist, norm_sim = ortografix.evaluate(
            test_indexes, encoder, decoder, dataset.target_vocab,
            WITH_ATTENTION, dataset.max_seq_len)
        dists.append(dist)
        norm_dists.append(norm_dist)
        norm_sims.append(norm_sim)
    print('avg dist = {}'.format(stats.mean(dists)))
    if len(dists) > 1:
        print('std dist = {}'.format(stats.stdev(dists)))
    print('avg nsim = {}'.format(stats.mean(norm_sims)))
    if len(norm_sims) > 1:
        print('std nsim = {}'.format(stats.stdev(norm_sims)))
