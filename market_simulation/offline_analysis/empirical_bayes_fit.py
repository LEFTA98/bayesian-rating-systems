"""
class for fitting the empirical Bayes prior.
"""

import pandas as pd


class PriorFitter:

    def __init__(self, ratings_style):
        self.ratings_style = ratings_style

    def ingest_fit_data(self, fit_data: pd.DataFrame) -> None:
        pass

    def fit(self):
        pass

    def _fit_bernoulli_data(self):
        pass

    def _fit_categorical_data(self):
        pass

    def _fit_continuous_data(self):
        pass
