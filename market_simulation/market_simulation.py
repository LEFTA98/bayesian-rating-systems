"""
central class for running market simulations.  The Simulator object ingests data in the form of a Pandas DataFrame,
can fit an empirical Bayesian prior to the data it is given, and run simulations with various different prior strengths.
"""

from typing import Tuple

import pandas as pd
from joblib import Parallel, delayed

from offline_analysis.empirical_bayes_fit import PriorFitter
from offline_analysis.data_cleaner import DataCleaner
from offline_analysis.run_simulation import SimRunner, DEFAULT_SELECTION


class Simulator:

    def __init__(self, ratings_style: str, selection_style: str = DEFAULT_SELECTION):
        self.ratings_style = ratings_style
        self.data = None
        self.prior = None
        self.outputs = None
        self.prior_fitter = PriorFitter(ratings_style)
        self.data_cleaner = DataCleaner(ratings_style)
        self.simulation_runner = SimRunner(ratings_style, selection_style)

    def ingest_data(self, data: pd.DataFrame, product_col: str, ratings_col: str) -> None:
        pass

    def fit_eb_prior(self) -> Tuple[str, str]:
        pass

    def run_simulations(self, num_agents: int) -> None:
        if not self.prior:
            self.fit_eb_prior()

        pass

    def get_snapshot_data(self):
        pass

    def get_market_data(self):
        pass

    def get_simulation_data(self):
        pass