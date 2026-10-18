"""Helpers for evaluating fixed CART facility configurations."""

from contextlib import contextmanager


@contextmanager
def fixed_e1(instance, e1_dict):
    """Temporarily fix facility-opening variables to a binary candidate."""
    for facility, value in e1_dict.items():
        instance.E1[facility].fix(int(value))
    try:
        yield
    finally:
        for facility in instance.m:
            instance.E1[facility].unfix()
