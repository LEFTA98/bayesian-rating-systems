"""
central class for running market simulations.  The Simulator object ingests data in the form of a Pandas DataFrame,
can fit an empirical Bayesian prior to the data it is given, and run simulations with various different prior strengths.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from offline_analysis.prior_fitter import PriorFitter
from offline_analysis.data_cleaner import DataCleaner
from offline_analysis.run_simulation import SimRunner
import offline_analysis.run_simulation as sim

DEFAULT_PROCESSES = -1
DEFAULT_RATINGS = 'bernoulli'
DEFAULT_ETA_VALUES = (0.001, 1, 10, 100, 1000, 10000)


# TODO rename to something more informative
class Simulator:

    def __init__(self, ratings_style: str = DEFAULT_RATINGS, selection_style: str = sim.DEFAULT_SELECTION):
        self.ratings_style = ratings_style
        self.prior = None
        self.output_market_data = []
        self.output_snapshots = []
        self.output_market_history = []
        self._prior_fitter = PriorFitter(ratings_style)  # TODO privatize these
        self._data_cleaner = DataCleaner(ratings_style)
        self._simulation_runner = SimRunner(ratings_style, selection_style)

    def add_data(self, data: pd.DataFrame, product_col: str, ratings_col: str) -> None:
        self._data_cleaner.ingest(data, product_col, ratings_col)

    def fit_prior(self) -> Tuple[float, ...]:
        priors = self._prior_fitter.fit(self._data_cleaner.train_quality)
        self.prior = priors
        return priors

    #TODO Figure out how to add customizeable parameters here
    def run_simulations(self,
                        num_threads: int = DEFAULT_PROCESSES,
                        etas: Tuple[float, ...] = DEFAULT_ETA_VALUES,
                        timesteps: int = sim.DEFAULT_NUM_TIMESTEPS,
                        rho: float = sim.DEFAULT_RHO,
                        mkt_size: int = sim.DEFAULT_MKT_SIZE,
                        num_users: int = sim.DEFAULT_NUM_USERS,
                        rng: Union[None, int] = None) -> None:
        if not self.prior:
            self.fit_prior()

        input_df = self._data_cleaner.test_data
        products = list(self._data_cleaner.test_quality.keys())

        prior_array = np.expand_dims(np.array(self.prior), 1)
        etas_array = np.expand_dims(np.array(etas), 1)

        priors_to_test = (prior_array @ etas_array.T).T
        prior_names = [str(eta) for eta in etas]

        if num_threads == 1:
            data, snapshots, market_histories = self._simulation_runner.run_single_simulation(input_df,
                                                                                              products,
                                                                                              priors_to_test[0],
                                                                                              timesteps=timesteps,
                                                                                              rho=rho,
                                                                                              mkt_size=mkt_size,
                                                                                              num_users=num_users,
                                                                                              seed=rng,
                                                                                              id_name=prior_names[0])
            self.output_market_data.append(data)
            self.output_snapshots.append(snapshots)
            self.output_market_history.append(market_histories)
        else:
            parallel = Parallel(n_jobs=min(len(etas), num_threads), verbose=10)
            result_data = parallel(delayed(self._simulation_runner.run_single_simulation)(input_df,
                                                                                          products,
                                                                                          priors_to_test[i],
                                                                                          timesteps=timesteps,
                                                                                          rho=rho,
                                                                                          mkt_size=mkt_size,
                                                                                          num_users=num_users,
                                                                                          seed=rng,
                                                                                          id_name=prior_names[i])
                                   for i in range(len(prior_names)))
            result_data = list(result_data)
            for data, snapshots, market_histories in result_data:
                self.output_market_data.append(data)
                self.output_snapshots.append(snapshots)
                self.output_market_history.append(market_histories)


    def get_snapshot_data(self):
        pass

    def get_market_data(self):
        pass

    def get_simulation_data(self):
        pass
