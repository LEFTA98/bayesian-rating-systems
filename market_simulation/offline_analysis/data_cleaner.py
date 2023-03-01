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
        self.products_list = None

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

        df = copy.deepcopy(data)
        df['dummy'] = 1
        df = df.pivot_table(index=product_col, columns=ratings_col, values='dummy', aggfunc=np.sum)
        df = df.fillna(0)

        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            df[ratings_col] = df[1]/(df[0] + df[1])

        self.train_quality, self.test_quality = train_test_split(df, test_size=split, random_state=rng)
        self.test_data = data[data[product_col].isin(list(self.test_quality.index))]
        self.product_col = product_col
        self.ratings_col = ratings_col
        self.products_list = list(set(data[product_col]))

    def upsample(self, num_samples: int) -> None:
        percentiles = np.linspace(0, 100, num_samples + 1) / 100

        df = copy.deepcopy(self.test_quality).drop_duplicates()

        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            df = df.drop(columns=[self.ratings_col])

        df['ranking_quality'] = df.to_numpy() @ self.weights/np.sum(df.to_numpy(), axis=1)
        df = df.sort_values(by='ranking_quality', axis=0)
        percentiles = np.round(percentiles * len(df)).astype(int)
        percentiles = percentiles[:-1]  # 100th percentile doesn't exist omit it
        sampled_products = list(df.iloc[percentiles].index)
        self.test_data = self.test_data[self.test_data[self.product_col].isin(sampled_products)]
        self.products_list = sampled_products
