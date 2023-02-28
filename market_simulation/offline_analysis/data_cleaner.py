"""class that cleans raw data input to the Simulator class."""

from typing import Union
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BERNOULLI_RATINGS_NAME = "bernoulli"
CATEGORICAL_RATINGS_NAME = "categorical"


def _convert_to_bernoulli(data: pd.DataFrame, ratings_col: str):
    data[ratings_col] = (data[ratings_col] > 0.5).astype(int)
    return data


class DataCleaner:

    def __init__(self, ratings_style: str):
        self.ratings_style = ratings_style
        self.raw_data = None
        self.product_col = None
        self.ratings_col = None
        self.train_quality = None
        self.test_quality = None
        self.test_data = None
        self.weights = None

    def ingest(self,
               data: pd.DataFrame,
               product_col: str,
               ratings_col: str,
               split: float = 0.4,
               rng: Union[None, int] = None) -> None:
        self.raw_data = copy.deepcopy(data)

        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            data = _convert_to_bernoulli(data, ratings_col)
            self.weights = np.array([0, 1])
        elif self.ratings_style == CATEGORICAL_RATINGS_NAME:
            self.weights = np.sort(np.unique(data[ratings_col].values))

        df = data.groupby([product_col]).mean(numeric_only=True)[ratings_col]
        self.train_quality, self.test_quality = train_test_split(df, test_size=split, random_state=rng)
        self.test_data = data[data[product_col].isin(list(self.test_quality.keys()))]
        self.product_col = product_col
        self.ratings_col = ratings_col

    def upsample(self, num_samples: int) -> None:
        percentiles = np.linspace(0, 100, num_samples + 1) / 100

        df = self.test_quality.drop_duplicates().sort_values(by=self.product_col)
        percentiles = np.round(percentiles * len(df)).astype(int)
        percentiles = percentiles[:-1]  # 100th percentile doesn't exist omit it
        sampled_products = list(df.iloc[percentiles]['video_id'])
        self.test_data = self.test_data[self.test_data['video_id'].isin(sampled_products)]
