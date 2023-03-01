"""
class for fitting the empirical Bayes prior.
"""

from typing import Tuple, Union

import pandas as pd
import numpy as np
from scipy.stats import beta
import dirichlet as di

from .data_cleaner import BERNOULLI_RATINGS_NAME, CATEGORICAL_RATINGS_NAME


def bernoulli_zero_inflated_adjust(i, n, prior=1 / 2):
    return (i * (n - 1) + prior) / n


def _fit_bernoulli_data(fit_data: pd.DataFrame, ratings_col: str) -> Tuple[float, float]:
    data = fit_data[ratings_col]
    prior_a, prior_b, loc, scale = beta.fit([bernoulli_zero_inflated_adjust(i, len(data.values), 1 / 2)
                                             for i in data.values], floc=0, fscale=1)
    return prior_b, prior_a


def _fit_categorical_data(fit_data: pd.DataFrame, n_iter: int):

    data = np.divide(fit_data.to_numpy(), np.array([np.sum(fit_data.to_numpy(), axis=1)
                                                   for _ in range(fit_data.to_numpy().shape[1])]).T)
    data = data[np.product(data, axis=1) != 0]

    return di.mle(data, method='fixedpoint', maxiter=n_iter)


class PriorFitter:

    def __init__(self, ratings_style):
        self.ratings_style = ratings_style

    #TODO fix the parameters here and their typings
    def fit(self, fit_data: Union[pd.DataFrame, pd.Series], ratings_col: str) -> Tuple[float, ...]:
        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            return _fit_bernoulli_data(fit_data, ratings_col)
        elif self.ratings_style == CATEGORICAL_RATINGS_NAME:
            return _fit_categorical_data(fit_data, n_iter=1000)

    def _fit_continuous_data(self, fit_data: pd.DataFrame):
        pass
