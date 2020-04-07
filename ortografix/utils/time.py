"""Time utils."""
import time
import math

__all__ = ('time_since')


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {}s'.format(m, round(s))


def time_since(since, percent):
    now = time.time()
    elapsed = now - since
    ratio = elapsed / percent
    remaining = ratio - elapsed
    return '{} (- {})'.format(as_minutes(elapsed), as_minutes(remaining))
