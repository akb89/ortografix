"""Unit and behavioral tests for preprocessing."""

import ortografix.utils.preprocessing as preputils

def test_prepare_token_based_source_target_dict():
    input_stream = ['This is a test .\tCeci est un test .',
                    'It contains multiple lines .\tIl contient plusieurs lignes .']
    source_dict, target_dict = preputils._prepare_source_target_dict(
        input_stream, character_based=False)
    assert len(source_dict) == len(target_dict)
    assert len(source_dict) == 11

def test_prepare_character_based_source_target_dict():
    input_stream = ['This is a test .\tCeci est un test .',
                    'It contains multiple lines .\tIl contient plusieurs lignes .']
    source_dict, target_dict = preputils._prepare_source_target_dict(
        input_stream, character_based=True)
    print(source_dict)
    print(target_dict)
    assert len(source_dict) == 18
    assert len(target_dict) == 17
