"""Helper functions that implements multi-armed bandit-style selection functions for choice functions."""

import numpy as np


def random_argmax(alist, rng=None):
    if not rng:
        rng = np.random.default_rng()

    maxval = max(alist)
    argmax = [idx for idx, val in enumerate(alist) if val == maxval]
    return rng.choice(argmax)


def ts_action(actions, num_success, num_failure, rng=None):
    if not rng:
        rng = np.random.default_rng()

    p_hat = [rng.beta(num_success[a], num_failure[a]) for a in actions]
    a = random_argmax(p_hat, rng)
    return a