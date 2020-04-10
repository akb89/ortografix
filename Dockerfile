FROM python:3

ADD . /ortografix

WORKDIR /ortografix

RUN python3 setup.py install
