# ortografix
[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][build-image]][build-url]
[![MIT License][license-image]][license-url]


[release-image]:https://img.shields.io/github/release/akb89/ortografix.svg?style=flat-square
[release-url]:https://github.com/akb89/ortografix/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/ortografix.svg?style=flat-square
[pypi-url]:https://pypi.org/project/ortografix/
[build-image]:https://img.shields.io/github/workflow/status/akb89/ortografix/CI?style=flat-square
[build-url]:https://github.com/akb89/ortografix/actions?query=workflow%3ACI
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt

Welcome to ortografix, a seq2seq model for automatic ortografic simplification, coded with pytorch 1.4.

## Install
via pip:
```shell
pip3 install ortografix
```
or, after a git clone:
```shell
python3 setup.py install
```

## Train
To train a model, run:
```shell
ortografix train \
--data /abs/path/to/training/data \
--model-type gru \
--shuffle \
--hidden-size 256 \
--num-layers 1 \
--bias \
--dropout 0 \
--learning-rate 0.01 \
--epochs 10 \
--print-every 100 \
--use-teacher-forcing \
--teacher-forcing-ratio 0.5 \
--output-dirpath /abs/path/to/output/directory/whereto/save/model \
--with-attention \
--character-based
```

## Test
### Qualitative evaluation
To qualitatively evaluate the output of the model on a set of 10 randomly selected sentences from a given dev/test set, run:
```shell
ortografix evaluate \
--data /abs/path/to/test/data.txt \
--model /abs/path/to/model/directory/ \
--random 10
```
### Quantitative evaluation
To quantitatively evaluate the output of the model on a given dev/test set, run:
```shell
ortografix evaluate \
--data /abs/path/to/test/data.txt \
--model /abs/path/to/model/directory
```
Quantitative evaluation will return:
1. The sum of all edit (Levenshtein) distance computed across all test pairs
2. The average edit distance computed across all test pairs
3. The average normalized edit distance
4. The average normalized edit similarity

All measure are computed via [textdistance](https://github.com/life4/textdistance).
