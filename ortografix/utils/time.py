"""Time utils."""
import time
import math

__all__ = ('time_since')


def as_minutes(seconds):
    """Convert input to minutes."""
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '{}m {}s'.format(minutes, round(seconds))


def time_since(since, percent):
    """Return time since."""
    now = time.time()
    elapsed = now - since
    ratio = elapsed / percent
    remaining = ratio - elapsed
    return '{} (- {})'.format(as_minutes(elapsed), as_minutes(remaining))
