"""Unit and behavioral tests for preprocessing."""

import ortografix.utils.processing as putils


def test_index_sequence():
    sequence = ['this', 'is', 'a', 'test']
    char2idx = {'t': 3, 'h': 4, 'i': 5, 's': 6, 'a': 7, 'UNK': 8}
    indexes = putils.index_sequence(sequence, char2idx, sos_idx=0, sep_idx=2,
                                    eos_idx=1, unk='UNK')
    assert indexes == [0, 3, 4, 5, 6, 2, 5, 6, 2, 7, 2, 3, 8, 6, 3, 1]
