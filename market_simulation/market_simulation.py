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

from dash import Dash, dcc, html, Input, Output
from jupyter_dash import JupyterDash
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
        self.dataframes_generated = []
        self._prior_fitter = None
        self._data_cleaner = None
        self._simulation_runner = None


    #TODO add warning here after ingestion that maybe user should consider upsampling
    def add_data(self,
                 data: pd.DataFrame,
                 product_col: str,
                 ratings_col: str,
                 ratings_style: Union[None, str] = None,
                 split: float = 0.4,
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

        self._data_cleaner.ingest(data, product_col, ratings_col, split=split, cutoff=cutoff, rng=rng)

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
                        timesteps: Union[int, None] = None,
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
        products = self._data_cleaner.test_products_list
        weights = self._data_cleaner.weights
        product_col = self._data_cleaner.product_col
        rating_col = self._data_cleaner.ratings_col

        n_unique_prods = list(input_df[product_col].unique())

        if not timesteps:
            timesteps = int(100 * 1/rho * len(n_unique_prods))
            print(f"defaulting to {timesteps} number of timesteps")

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
                                                                                              id_name=prior_names[i],
                                                                                              show_tqdm=False)
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

    def gen_new_data(self,
                     num_threads: int = DEFAULT_PROCESSES,
                     etas: Tuple[float, ...] = DEFAULT_ETA_VALUES,
                     timesteps: Union[int, None] = None,
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
        products = self._data_cleaner.test_products_list
        weights = self._data_cleaner.weights
        product_col = self._data_cleaner.product_col
        rating_col = self._data_cleaner.ratings_col

        n_unique_prods = list(input_df[product_col].unique())

        if not timesteps:
            timesteps = int(100 * 1/rho * len(n_unique_prods))
            print(f"defaulting to {timesteps} number of timesteps")

        prior_array = np.expand_dims(np.array(self.prior), 1)
        etas_array = np.expand_dims(np.array(etas), 1)

        priors_to_test = (prior_array @ etas_array.T).T
        prior_names = [str(eta) for eta in etas]

        if num_threads == 1:
            data, gen_data, market_histories = \
                self._simulation_runner.generate_new_data_from_simulations(input_df,
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
            gen_data['eta'] = prior_names[0]
            self.dataframes_generated.append(gen_data)
            self.output_market_history.append(market_histories)

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
        df['agg_rating'] = df.to_numpy()/np.repeat(df.to_numpy().sum(axis=1)[:, np.newaxis],
                                                   df.shape[1], axis=1) @ self._data_cleaner.weights
        qual_dict = df['agg_rating'].to_dict()
        qfunc = np.vectorize(lambda x: qual_dict[x])

        md, mh = self.get_simulation_data(), self.get_market_data()

        to_plot = []

        for i in range(len(md)):
            market_hist = qfunc(mh[i])
            regret = np.sum(np.max(market_hist[:, :-1], axis=1) - market_hist[:, -1])/market_hist.shape[0]
            std = get_average_plays_std(md[i])
            eta = list(md[i].keys())[0][1]

            to_plot.append(pd.DataFrame([[eta, std, regret]], columns=['etas', 'avg standard dev in play %', 'regret']))

        to_plot_df = pd.concat(to_plot)
        x = list(to_plot_df['avg standard dev in play %'])
        y = list(to_plot_df['regret'])
        labels = to_plot_df['etas'].to_numpy()

        fig, ax = plt.subplots()
        fig.set_size_inches(18, 6)
        ax.scatter(x, y)
        plt.xlabel('Producer Consistency (average standard deviation in play %)')
        plt.ylabel('Consumer Efficiency (per-timestep regret)')

        for i in range(labels.shape[0]):
            ax.annotate(f"$\\eta$={labels[i]}", (x[i], y[i]))

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
        f.set_size_inches(3*len(keys), 3)

        for j in range(len(keys)):
            plt.xlim((0, 1))
            plot_data = {'bin': [], 'val': []}
            if len(keys) > 1:
                plt.sca(axes[j])

            for i in range(len(plays_data[keys[j]])):
                for item in plays_data[keys[j]][i]:
                    plot_data['bin'].append(i + 1)
                    plot_data['val'].append(item)

            plt.title(f"$\\eta$={keys[j]}")
            if j == 0:
                ylab = "true quality quartile"
            else:
                ylab = None

            xlab = 'play ratio $P(l)$'

            plot_data = pd.DataFrame(plot_data)
            g = sns.violinplot(y=plot_data['bin'], x=plot_data['val'], inner=None, orient='h', palette='colorblind')
            if j != 0:
                g.set(yticklabels=[], yticks=[])

            plt_object = axes if len(keys) == 1 else axes[j]
            plt_object.scatter(x=plot_data.groupby('bin').mean()['val'].values,
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

    def _get_pareto_frontier_data(self):
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
        df['agg_rating'] = df.to_numpy()/np.repeat(df.to_numpy().sum(axis=1)[:, np.newaxis],
                                                   df.shape[1], axis=1) @ self._data_cleaner.weights
        qual_dict = df['agg_rating'].to_dict()
        qfunc = np.vectorize(lambda x: qual_dict[x])

        md, ss, mh = self.output_market_data, self.output_snapshots, self.output_market_history

        to_plot = []

        for i in range(len(md)):
            market_hist = qfunc(mh[i])
            regret = np.sum(np.max(market_hist[:, :-1], axis=1) - market_hist[:, -1])/market_hist.shape[0]
            std = get_average_plays_std(md[i])
            eta = list(md[i].keys())[0][1]

            to_plot.append(pd.DataFrame([[eta, std, regret]], columns=['etas', 'avg standard dev in play %', 'regret']))

        to_plot_df = pd.concat(to_plot)
        x = list(to_plot_df['avg standard dev in play %'])
        y = list(to_plot_df['regret'])
        labels = to_plot_df['etas'].to_numpy()

        return x, y, labels

    def _get_interactive_data(self):
        md = self.get_simulation_data()

        df = copy.deepcopy(self._data_cleaner.test_quality.loc[self._data_cleaner.products_list])

        if self.ratings_style == BERNOULLI_RATINGS_NAME:
            df = df.drop(columns=[self._data_cleaner.ratings_col])
        df['agg_rating'] = df.to_numpy()/np.repeat(df.to_numpy().sum(axis=1)[:, np.newaxis],
                                                   df.shape[1], axis=1) @ self._data_cleaner.weights
        qual_dict = df['agg_rating'].to_dict()

        df_to_return = {'eta': [], 'prod': [], 'true quality': [], 'play ratio': []}

        for d in md:
            eta = list(d.keys())[0][1]
            for item in d:
                prod = item[0]
                tq = qual_dict[prod]
                for lifespan in d[item]:
                    if lifespan.shape[0] > 1:
                        pr = (np.sum(lifespan[-1]) - np.sum(lifespan[0])) / (lifespan.shape[0] - 1)

                        df_to_return['eta'].append(eta)
                        df_to_return['prod'].append(prod)
                        df_to_return['true quality'].append(tq)
                        df_to_return['play ratio'].append(pr)

        return pd.DataFrame(df_to_return)

    def get_simulation_level_interactive_view(self, notebook=False):
        """
        Creates an interactive Dash figure to explore simulation results at a prior strength-level, showing a histogram
        describing the distribution of the "play ratio" of products, i.e. the percentage of time a product was bought
        during its lifetime in the market. The drop-down menu filters specifically for the simulation with the given
        prior strength, while the slider filters on the true quality of products shown.

        :param notebook: boolean representing whether or not Dash graph should be output to notebook.
        :return: NoneType
        """
        df = self._get_interactive_data()

        df['rounded true quality'] = df['true quality'].round()
        df['rounded play ratio'] = (df['play ratio'] * 20).round() / 20
        df = df.groupby(['eta', 'rounded true quality', 'rounded play ratio']).count().reset_index()[
            ['eta', 'rounded true quality', 'rounded play ratio', 'play ratio']]

        eta_list = list(df['eta'].unique())
        dfs_to_plot = {}

        for eta in eta_list:
            df_to_plot = df[df['eta'] == eta]
            to_add = []

            for val in list(df['rounded play ratio'].unique()):
                for rtq in list(df_to_plot['rounded true quality'].unique()):
                    if val not in df_to_plot[df_to_plot['rounded true quality'] == rtq]['rounded play ratio']:
                        temp_df = pd.DataFrame({'eta': eta,
                                                'rounded true quality': rtq,
                                                'rounded play ratio': val,
                                                'play ratio': 0}, index=[0])
                        to_add.append(temp_df)

            df_to_plot = pd.concat(to_add + [df_to_plot])

            dfs_to_plot[eta] = df_to_plot

        slider_min, slider_max = df['rounded true quality'].min(), df['rounded true quality'].max()

        df_to_plot = copy.deepcopy(dfs_to_plot[eta_list[0]])
        df_to_plot = df_to_plot.sort_values('rounded true quality')
        df_to_plot['rounded true quality'] = df_to_plot['rounded true quality'].astype(str)

        cdmap = {k: v for k, v in zip(list(df_to_plot['rounded true quality'].unique()),
                                      px.colors.qualitative.Alphabet[
                                      :len(df_to_plot['rounded true quality'].unique())])}

        fig = px.bar(df_to_plot,
                     x='rounded play ratio',
                     y='play ratio',
                     color='rounded true quality',
                     color_discrete_map=copy.deepcopy(cdmap))

        fig.update_xaxes(title='play ratio')
        fig.update_yaxes(title='count')
        fig.update_layout(barmode='stack')

        app = JupyterDash() if notebook else Dash()

        app.layout = html.Div([
            html.Div([
                "Prior Strength (eta)",
                dcc.Dropdown(
                    eta_list,
                    eta_list[0],
                    id="eta_dropdown",
                ),
            ]),
            dcc.Graph(figure=fig, id='eta_level_view'),
            dcc.RangeSlider(slider_min, slider_max, 1, value=[slider_min, slider_max], id='rtq_slider')
        ])

        @app.callback(
            Output('eta_level_view', 'figure'),
            Input('eta_dropdown', 'value'),
            Input('rtq_slider', 'value'))
        def update_figure(value_dropdown, value_slider):

            dtp = copy.deepcopy(dfs_to_plot[value_dropdown])
            dtp = dtp[dtp['rounded true quality'].between(value_slider[0], value_slider[1])]
            dtp = dtp.sort_values('rounded true quality')
            dtp['rounded true quality'] = dtp['rounded true quality'].astype(str)

            f = px.bar(dtp,
                       x='rounded play ratio',
                       y='play ratio',
                       color='rounded true quality',
                       color_discrete_map=copy.deepcopy(cdmap))

            f.update_xaxes(title='play ratio')
            f.update_yaxes(title='count')
            f.update_layout(barmode='stack')

            return f

        if notebook:
            print('eta value')
            app.run_server(debug=True, use_reloader=False, mode="inline")
        else:
            app.run_server(debug=True, use_reloader=False)

    def get_product_level_interactive_view(self, notebook=False):
        """
        Creates an interactive Dash figure that explores simulation data at the product level. The bar graph describes
        for each selected product how many of each star rating it got over the selected simulations.

        :param notebook: boolean representing whether or not Dash graph should be output to notebook.
        :return: NoneType
        """
        weighting_data = []
        data_weights = self._data_cleaner.weights
        columns = ['eta', 'prod'] + ['c'+str(item) for item in list(data_weights)]

        data = self._get_interactive_data()[['eta', 'prod', 'true quality']].drop_duplicates()
        data2 = self.get_simulation_data()

        for datadict in data2:
            for (k1, k2) in datadict:
                agg_ratings = np.zeros(len(data_weights))
                for lifespan in datadict[(k1, k2)]:
                    agg_ratings += lifespan[-1] - lifespan[0]
                d = {k: v for k, v in zip(columns, [[k2], [k1]] + [[item] for item in list(np.round(agg_ratings))])}

                weighting_data.append(pd.DataFrame(d))

        weighting_data = pd.concat(weighting_data)

        data = data.merge(weighting_data, on=['eta', 'prod'])
        data = pd.wide_to_long(data, stubnames='c', i=['eta', 'prod', 'true quality'], j='rating').reset_index()
        data['true quality'] = data['true quality'].round(2)

        eta_dropdown_options = sorted(list(data['eta'].unique()))
        prod_dropdown_options = sorted(list(data['prod'].unique()))

        cdmap = {k: v for k, v in zip(list(data['eta'].unique()),
                                      px.colors.qualitative.Alphabet[:len(data['eta'].unique())])}

        mask = (data['eta'].isin(sorted([eta_dropdown_options[0]]))) & (
                    data['prod'].isin(sorted([prod_dropdown_options[0]])))

        fig = px.bar(data[mask],
                     x='rating',
                     y='c',
                     color='eta',
                     barmode='group',
                     hover_name='prod',
                     hover_data=['true quality'],
                     color_discrete_map=cdmap)

        fig.update_yaxes(title='count')

        app = JupyterDash() if notebook else Dash()

        app.layout = html.Div([
            dcc.Graph(figure=fig, id='product_level_view'),
            html.Div([
                "etas",
                dcc.Dropdown(
                    eta_dropdown_options,
                    [eta_dropdown_options[0]],
                    id="eta_dropdown",
                    multi=True
                )
            ]),
            html.Div([
                "products",
                dcc.Dropdown(
                    prod_dropdown_options,
                    [prod_dropdown_options[0]],
                    id="prod_dropdown",
                    multi=True
                ),
            ]),
        ])

        @app.callback(
            Output('product_level_view', 'figure'),
            Input('eta_dropdown', 'value'),
            Input('prod_dropdown', 'value'))
        def update_figure(value_etas, value_products):
            m = (data['eta'].isin(sorted(value_etas))) & (
                data['prod'].isin(sorted(value_products)))

            f = px.bar(data[m],
                       x='rating',
                       y='c',
                       color='eta',
                       barmode='group',
                       hover_name='prod',
                       hover_data=['true quality'],
                       color_discrete_map=cdmap)

            f.update_yaxes(title='count')

            return f

        if notebook:
            app.run_server(debug=True, use_reloader=False, mode="inline")
        else:
            app.run_server(debug=True, use_reloader=False)
