"""Helper functions that implements multi-armed bandit-style selection functions for choice functions."""

import numpy as np
from scipy.stats import dirichlet


def random_argmax(alist, rng=None):
    if not rng:
        rng = np.random.default_rng()

    maxval = max(alist)
    argmax = [idx for idx, val in enumerate(alist) if val == maxval]
    return rng.choice(argmax)


def ts_action(actions, weights, params, rng=None):
    if not rng:
        rng = np.random.default_rng()

    p_hat = np.array([dirichlet.rvs(params[a], 1, random_state=rng) for a in actions])
    p_hat = p_hat @ weights

    a = random_argmax(p_hat, rng)
    return a
