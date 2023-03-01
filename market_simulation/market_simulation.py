"""
central class for running market simulations.  The Simulator object ingests data in the form of a Pandas DataFrame,
can fit an empirical Bayesian prior to the data it is given, and run simulations with various different prior strengths.
"""

from typing import Tuple, Union
import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from offline_analysis.prior_fitter import PriorFitter
from offline_analysis.data_cleaner import DataCleaner, BERNOULLI_RATINGS_NAME
from offline_analysis.run_simulation import SimRunner
import offline_analysis.run_simulation as sim

DEFAULT_PROCESSES = -1
DEFAULT_RATINGS = BERNOULLI_RATINGS_NAME
DEFAULT_ETA_VALUES = (0.001, 1, 10, 100, 1000, 10000)


class BayesianRatingsManager:

    def __init__(self, ratings_style: str = DEFAULT_RATINGS, selection_style: str = sim.DEFAULT_SELECTION):
        self.ratings_style = ratings_style
        self.prior = None
        self.output_market_data = []
        self.output_snapshots = []
        self.output_market_history = []
        self._prior_fitter = PriorFitter(ratings_style)
        self._data_cleaner = DataCleaner(ratings_style)
        self._simulation_runner = SimRunner(ratings_style, selection_style)

    #TODO add warning here after ingestion that maybe user should consider upsampling
    def add_data(self, data: pd.DataFrame, product_col: str, ratings_col: str, rng: Union[None, int] = None) -> None:
        self._data_cleaner.ingest(data, product_col, ratings_col, rng=rng)

    def upsample(self, num_samples: int = 100):
        self._data_cleaner.upsample(num_samples)

    def fit_prior(self) -> Tuple[float, ...]:
        priors = self._prior_fitter.fit(self._data_cleaner.train_quality, self._data_cleaner.ratings_col)
        self.prior = priors
        return priors

    #TODO Figure out how to add customizeable parameters here
    #TODO make tqdm progress bars for each thing here
    def run_simulations(self,
                        num_threads: int = DEFAULT_PROCESSES,
                        etas: Tuple[float, ...] = DEFAULT_ETA_VALUES,
                        timesteps: int = sim.DEFAULT_NUM_TIMESTEPS,
                        rho: float = sim.DEFAULT_RHO,
                        mkt_size: int = sim.DEFAULT_MKT_SIZE,
                        num_users: int = sim.DEFAULT_NUM_USERS,
                        rng: Union[None, int] = None) -> None:
        if self.prior is None:
            self.fit_prior()

        input_df = self._data_cleaner.test_data
        products = self._data_cleaner.products_list
        weights = self._data_cleaner.weights
        product_col = self._data_cleaner.product_col
        rating_col = self._data_cleaner.ratings_col

        prior_array = np.expand_dims(np.array(self.prior), 1)
        etas_array = np.expand_dims(np.array(etas), 1)

        priors_to_test = (prior_array @ etas_array.T).T
        prior_names = [str(eta) for eta in etas]

        if num_threads == 1:
            data, snapshots, market_histories = self._simulation_runner.run_single_simulation(input_df,
                                                                                              products,
                                                                                              weights,
                                                                                              priors_to_test[0],
                                                                                              product_col,
                                                                                              rating_col,
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
                                                                                          weights,
                                                                                          priors_to_test[i],
                                                                                          product_col,
                                                                                          rating_col,
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
        return self.output_snapshots

    def get_market_data(self):
        return self.output_market_history

    def get_simulation_data(self):
        return self.output_market_data

    def clear_outputs(self):
        self.output_market_data, self.output_snapshots, self.output_market_history = [], [], []

    #TODO add other visualizations
    def summary(self, savefigs: bool = False) -> None:

        if savefigs:
            dir_path = 'saved_plots'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        def get_average_plays_std(datadict):
            a = []
            for item in datadict:
                l_array = []
                for lifespan in datadict[item]:
                    if lifespan.shape[0] > 1:
                        l_array.append((np.sum(lifespan[-1]) - np.sum(lifespan[0])) / (lifespan.shape[0] - 1))

                a.append(np.std(l_array))

            return np.mean(a)

        if np.prod([len(self.output_market_history), len(self.output_snapshots), len(self.output_market_data)]) == 0:
            raise Exception("simulation outputs are empty. Use run_simulations to populate")

        df = copy.deepcopy(self._data_cleaner.test_quality.loc[self._data_cleaner.products_list])

        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            df = df.drop(columns=[self._data_cleaner.ratings_col])
        df['agg_rating'] = df.to_numpy() @ self._data_cleaner.weights
        qual_dict = df['agg_rating'].to_dict()
        qfunc = np.vectorize(lambda x: qual_dict[x])

        md, ss, mh = self.output_market_data, self.output_snapshots, self.output_market_history

        to_plot = []

        for i in range(len(md)):
            market_hist = qfunc(mh[i])
            regret = np.sum(np.max(market_hist[:, :-1], axis=1) - market_hist[:, -1])
            std = get_average_plays_std(md[i])
            eta = list(md[i].keys())[0][1]

            to_plot.append(pd.DataFrame([[eta, std, regret]], columns=['etas', 'avg standard dev in play %', 'regret']))

        to_plot_df = pd.concat(to_plot)
        x = list(to_plot_df['avg standard dev in play %'])
        y = list(to_plot_df['regret'])
        labels = to_plot_df['etas'].to_numpy()

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        ax.scatter(x, y)
        plt.xlabel('avg standard deviation in play %')
        plt.ylabel('total regret')

        for i in range(labels.shape[0]):
            ax.annotate(f"{labels[i]}", (x[i], y[i]))

        if savefigs:
            plt.savefig(f"{dir_path}/{str(datetime.now())}+pareto curve.pdf")

        plt.show()
        plt.clf()
