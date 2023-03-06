"""
central class for running market simulations.  The Simulator object ingests data in the form of a Pandas DataFrame,
can fit an empirical Bayesian prior to the data it is given, and run simulations with various different prior strengths.
"""

from typing import Tuple, Union
import copy
import os
from datetime import datetime
import contextlib
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

from offline_analysis.prior_fitter import PriorFitter
from offline_analysis.data_cleaner import DataCleaner, BERNOULLI_RATINGS_NAME, CATEGORICAL_RATINGS_NAME
from offline_analysis.run_simulation import SimRunner
import offline_analysis.run_simulation as sim

DEFAULT_PROCESSES = -1
DEFAULT_RATINGS = BERNOULLI_RATINGS_NAME
DEFAULT_ETA_VALUES = (0.001, 1, 10, 100, 1000, 10000)

# helper function for tracking progress of parallel work, taken from:
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class BayesianRatingsManager:

    def __init__(self, selection_style: str = sim.DEFAULT_SELECTION):
        self.ratings_style = None
        self.prior = None
        self.selection_style = selection_style
        self.output_market_data = []
        self.output_snapshots = []
        self.output_market_history = []
        self._prior_fitter = None
        self._data_cleaner = None
        self._simulation_runner = None


    #TODO add warning here after ingestion that maybe user should consider upsampling
    def add_data(self,
                 data: pd.DataFrame,
                 product_col: str,
                 ratings_col: str,
                 ratings_style: Union[None, str] = None,
                 cutoff: Union[None, int] = None,
                 rng: Union[None, int] = None,
                 verbose: bool = True) -> None:
        if ratings_style is None:
            if verbose:
                print("auto-detecting ratings data style.")
            unique_ratings_vals = data[ratings_col].unique()
            if list(np.sort(unique_ratings_vals).astype(int)) == [0, 1]:
                if verbose:
                    print("detected ratings type: binary (0 or 1)")
                self.ratings_style = BERNOULLI_RATINGS_NAME
            else:
                if verbose:
                    print("detected ratings type: discrete")
                self.ratings_style = CATEGORICAL_RATINGS_NAME
        else:
            self.ratings_style = ratings_style

        self._prior_fitter = PriorFitter(self.ratings_style)
        self._data_cleaner = DataCleaner(self.ratings_style)
        self._simulation_runner = SimRunner(self.ratings_style, self.selection_style)

        self._data_cleaner.ingest(data, product_col, ratings_col, cutoff=cutoff, rng=rng)

    def upsample(self, num_samples: int = 100, verbose: bool = True):
        if self._data_cleaner is None:
            raise Exception("""No DataCleaner object is defined for this Bayesian Ratings system.
                               Have you added data yet?""")
        self._data_cleaner.upsample(num_samples)
        if verbose:
            print(f"upsampling task down to {num_samples} products complete")

    def fit_prior(self, verbose: bool = True) -> Tuple[float, ...]:
        if self._data_cleaner is None:
            raise Exception("""No DataCleaner object is defined for this Bayesian Ratings system.
                               Have you added data yet?""")
        if self._prior_fitter is None:
            raise Exception("""No PriorFitter object is defined for this Bayesian Ratings system.
                               Have you added data yet?""")
        priors = self._prior_fitter.fit(self._data_cleaner.train_quality, self._data_cleaner.ratings_col)
        self.prior = priors
        if verbose:
            print("prior fit complete!")
            if self.ratings_style == BERNOULLI_RATINGS_NAME:
                print("prior type: Beta")
            elif self.ratings_style == CATEGORICAL_RATINGS_NAME:
                print(f"prior type: Dirichlet, with weights {self._data_cleaner.weights}")
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
                        rng: Union[None, int] = None,
                        verbose: bool = True) -> None:
        if self._data_cleaner is None:
            raise Exception("""No DataCleaner object is defined for this Bayesian Ratings system.
                               Have you added data yet?""")
        if self._prior_fitter is None:
            raise Exception("""No PriorFitter object is defined for this Bayesian Ratings system.
                               Have you added data yet?""")
        if self._simulation_runner is None:
            raise Exception("""No SimulationRunner object is defined for this Bayesian Ratings system.
                               Have you added data yet?""")

        if self.prior is None:
            self.fit_prior()

        input_df = self._data_cleaner.test_data
        products = self._data_cleaner.products_list
        weights = self._data_cleaner.weights
        product_col = self._data_cleaner.product_col
        rating_col = self._data_cleaner.ratings_col

        n_unique_prods = list(input_df[product_col].unique())

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

            covered_products = np.unique(market_histories[:, :-1])

            if covered_products.shape[0] < len(n_unique_prods):
                warnings.warn("WARNING: Not all products in universe appeared in simulation.")
        else:
            with tqdm_joblib(tqdm(desc="Simulation progress", total=len(prior_names))) as progress_bar:
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

                covered_products = np.unique(market_histories[:, :-1])
                eta = list(data.keys())[0][1]

                if covered_products.shape[0] < len(n_unique_prods):
                    warnings.warn(f"WARNING: Not all products in universe appeared in simulation for eta={eta}")

        if verbose:
            print("Simulations complete.")

    def get_snapshot_data(self):
        if not self.output_snapshots:
            raise Exception("No output data.")
        return self.output_snapshots

    def get_market_data(self):
        if not self.output_market_history:
            raise Exception("No output data.")
        return self.output_market_history

    def get_simulation_data(self):
        if not self.output_market_data:
            raise Exception("No output data.")
        return self.output_market_data

    def clear_outputs(self, verbose: bool = True):
        if verbose:
            print("outputs cleared.")
        self.output_market_data, self.output_snapshots, self.output_market_history = [], [], []

    def _show_pareto_frontier(self, savefigs: bool, dir_path: str):
        def get_average_plays_std(datadict):
            a = []
            for item in datadict:
                l_array = []
                for lifespan in datadict[item]:
                    if lifespan.shape[0] > 1:
                        l_array.append((np.sum(lifespan[-1]) - np.sum(lifespan[0])) / (lifespan.shape[0] - 1))

                a.append(np.std(l_array))

            return np.mean(a)
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

    def _show_by_quality_plot(self, savefigs: bool, dir_path: str, n_quants: int = 4, cutoff: int = 0):
        def get_binned_plays_array(datadict, video_df, num_quants, cutoff=0):
            p_data = [[] for _ in range(num_quants)]
            quantile_df = pd.qcut(video_df, np.linspace(0, 1, num_quants + 1), labels=range(num_quants))

            for product in datadict:
                quant = p_data[quantile_df[product[0]]]

                for lifespan in datadict[product]:
                    if lifespan.shape[0] > cutoff:
                        final_score = lifespan[-1]
                        quant.append((np.sum(final_score) - np.sum(lifespan[0])) / lifespan.shape[0])

            return p_data

        df = copy.deepcopy(self._data_cleaner.test_quality.loc[self._data_cleaner.products_list])

        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            df = df['liked']
        else:
            df['avg score'] = df.to_numpy()/np.repeat(df.to_numpy().sum(axis=1)[:, np.newaxis],
                                                      df.shape[1], axis=1) @ self._data_cleaner.weights
            df = df['avg score']
        plays_data = {}
        data_dict = self.get_simulation_data()
        keys = [list(k.keys())[0][1] for k in data_dict]
        for i in range(len(keys)):
            plays_data[keys[i]] = get_binned_plays_array(data_dict[i], df, n_quants, cutoff=cutoff)

        f, axes = plt.subplots(1, len(keys))
        f.set_size_inches(4*len(keys), 4)

        for j in range(len(keys)):
            plt.xlim((0, 1))
            plot_data = {'bin': [], 'val': []}
            plt.sca(axes[j])

            for i in range(len(plays_data[keys[j]])):
                for item in plays_data[keys[j]][i]:
                    plot_data['bin'].append(i + 1)
                    plot_data['val'].append(item)

            plt.title(keys[j])
            if j == 0:
                ylab = "true quality quartile"
            else:
                ylab = None

            xlab = 'play ratio $PR(v)$'

            plot_data = pd.DataFrame(plot_data)
            g = sns.violinplot(y=plot_data['bin'], x=plot_data['val'], inner=None, orient='h', palette='colorblind')
            if j != 0:
                g.set(yticklabels=[], yticks=[])
            axes[j].scatter(x=plot_data.groupby('bin').mean()['val'].values,
                        y=plot_data.groupby('bin').mean()['val'].index - 1,
                        color='white', s=100, edgecolors='black')
            g.set(ylabel=ylab, xlabel=xlab)

        if savefigs:
            plt.savefig(f"{dir_path}/{str(datetime.now())}+per-quality-plot.pdf")

        plt.show()
        plt.clf()

    #TODO add other visualizations
    def summary(self, savefigs: bool = False) -> None:

        if np.prod([len(self.output_market_history), len(self.output_snapshots), len(self.output_market_data)]) == 0:
            raise Exception("simulation outputs are empty. Use run_simulations to populate")

        dir_path = 'saved_plots'
        if savefigs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self._show_pareto_frontier(savefigs=savefigs, dir_path=dir_path)

        self._show_by_quality_plot(savefigs=savefigs, dir_path=dir_path)
