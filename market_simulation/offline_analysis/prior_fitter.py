"""
class for fitting the empirical Bayes prior.
"""

from typing import Tuple

import pandas as pd
import numpy as np
from scipy.stats import beta
from scipy.special import polygamma

from .data_cleaner import BERNOULLI_RATINGS_NAME, CATEGORICAL_RATINGS_NAME


def bernoulli_zero_inflated_adjust(i, n, prior=1/2):
    return (i*(n-1)+prior)/n


def _fit_bernoulli_data(fit_data: pd.Series) -> Tuple[float, float]:
    prior_a, prior_b, loc, scale = beta.fit([bernoulli_zero_inflated_adjust(i, len(fit_data.values), 1 / 2)
                                             for i in fit_data.values], floc=0, fscale=1)
    return prior_a, prior_b


def _fit_categorical_data(fit_data: pd.Series, weights: np.ndarray, n_iter: int = 1000):

    # helper function that uses newton iterations to compute inverse digamma
    def igamma(x: np.ndarray, newton_iters: int = 5) -> float:
        m = (x >= -2.22).astype(int)
        y = m * (np.exp(x) + 0.5) + (1 - m) * (-1 / (x - polygamma(0, 1)))

        for _ in range(newton_iters):
            y -= (polygamma(0, y) - x) / polygamma(1, y)

        return y

    alpha = np.ones(weights.shape)
    p = fit_data.value_counts().sort_index().values / len(fit_data)

    for _ in range(n_iter):
        last_alpha = alpha
        alpha = igamma(polygamma(0, np.sum(last_alpha)) + np.log(p))

    #TODO turn this into a warning
    if np.sum(np.abs(last_alpha - alpha)) > 1e-2:
        print("WARNING: MLE for dirichlet distribution did not converge")
    return alpha


class PriorFitter:

    def __init__(self, ratings_style):
        self.ratings_style = ratings_style

    def fit(self, fit_data: pd.DataFrame, weights: np.ndarray) -> Tuple[float, ...]:
        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            return _fit_bernoulli_data(fit_data)
        elif self.ratings_style == CATEGORICAL_RATINGS_NAME:
            return _fit_categorical_data(fit_data, weights)

    def _fit_continuous_data(self, fit_data: pd.DataFrame):
        pass
