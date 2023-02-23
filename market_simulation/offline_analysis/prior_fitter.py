"""
class for fitting the empirical Bayes prior.
"""

from typing import Tuple

import pandas as pd
from scipy.stats import beta

from .data_cleaner import BERNOULLI_RATINGS_NAME, CATEGORIAL_RATINGS_NAME


def bernoulli_zero_inflated_adjust(i, n, prior=1/2):
    return (i*(n-1)+prior)/n


class PriorFitter:

    def __init__(self, ratings_style):
        self.ratings_style = ratings_style

    def fit(self, fit_data: pd.DataFrame) -> Tuple[float, ...]:
        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            return self._fit_bernoulli_data(fit_data)
        elif self.ratings_style == CATEGORIAL_RATINGS_NAME:
            return self._fit_categorical_data(fit_data)

    def _fit_bernoulli_data(self, fit_data: pd.DataFrame) -> Tuple[float, float]:
        prior_a, prior_b, loc, scale = beta.fit([bernoulli_zero_inflated_adjust(i, len(fit_data.values), 1 / 2)
                                                 for i in fit_data.values], floc=0, fscale=1)
        return prior_a, prior_b

    def _fit_categorical_data(self, fit_data: pd.DataFrame):
        pass

    def _fit_continuous_data(self, fit_data: pd.DataFrame):
        pass
