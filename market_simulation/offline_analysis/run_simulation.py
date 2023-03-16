"""
class for actually running the simulations.
"""

from typing import List, Union
import copy

import pandas as pd
import numpy as np
from tqdm import tqdm

from .product_utils import ProductHelper, sample_chosen_df
from .choice_models.multi_armed_bandit import ts_action

DEFAULT_SELECTION = "thompson_sampling"
DEFAULT_NUM_TIMESTEPS = 5000000
DEFAULT_RHO = 0.01
DEFAULT_MKT_SIZE = 5
DEFAULT_NUM_USERS = 1
DEFAULT_SNAPSHOT_PROB = 0.001

DEFAULT_CHOICE_FN_MAPPING = {("bernoulli", "thompson_sampling"): ts_action,
                             ("categorical", "thompson_sampling"): ts_action}


class SimRunner:

    def __init__(self, ratings_style: str, selection_style: str = DEFAULT_SELECTION):
        self.ratings_style = ratings_style
        self.selection_style = selection_style

    def run_single_simulation(self,
                              input_df: pd.DataFrame,
                              products: List[str],
                              weights: np.ndarray,
                              priors: np.ndarray,
                              product_col: str,
                              rating_col: str,
                              timesteps: int = DEFAULT_NUM_TIMESTEPS,
                              rho: float = DEFAULT_RHO,
                              mkt_size: int = DEFAULT_MKT_SIZE,
                              num_users: int = DEFAULT_NUM_USERS,
                              snapshot_start: Union[bool, int] = None,
                              snapshotting_prob: int = DEFAULT_SNAPSHOT_PROB,
                              seed: Union[bool, int] = None,
                              id_name: Union[bool, str] = None,
                              show_tqdm: bool = True):
        rng = np.random.default_rng(seed)
        sampling_action = DEFAULT_CHOICE_FN_MAPPING[(self.ratings_style, self.selection_style)]
        if not snapshot_start:
            snapshot_start = timesteps // 5
        product_data = dict(zip(products, [[] for _ in range(len(products))]))
        priors_list = [copy.deepcopy(priors) for _ in range(len(products))]
        priors_dict = dict(zip(products, priors_list))
        snapshot_dict = dict()
        snapshot_num = 1

        curr_vids = np.array(list(rng.choice(products, mkt_size, replace=False)))
        remaining_vids = set(products).difference(set(curr_vids))

        helper = ProductHelper(product_data, curr_vids, list(curr_vids), priors_dict)

        unique_ratings_vals = list(set(input_df[rating_col].values))
        like_to_slot_dict = {k: v for k, v in zip(sorted(unique_ratings_vals), range(len(unique_ratings_vals)))}
        market_history = []

        if show_tqdm:
            range_to_iterate = tqdm(range(timesteps))
        else:
            range_to_iterate = range(timesteps)

        for t in range_to_iterate:
            market_history.append(copy.deepcopy(helper.mkt_ids))
            latest_sims = np.array([item[-1] for item in helper.market])
            actions = range(mkt_size)
            for m in range(num_users):
                a = sampling_action(actions, weights, latest_sims, rng=None)
                chosen_action_global_index = products.index(helper.mkt_ids[a])
                market_history[-1].append(copy.deepcopy(helper.mkt_ids[a]))
                like = sample_chosen_df(products,
                                        input_df,
                                        chosen_action_global_index,
                                        product_col,
                                        rating_col,
                                        rng=None)

                slot = like_to_slot_dict[like]

                # update prior
                helper.pull_arm_update_market(a, slot)

            # replenish the indices
            flips = rng.binomial(1, rho, mkt_size)

            draws = rng.choice(sorted(list(remaining_vids)) +
                               [helper.mkt_ids[i] for i in range(len(helper.mkt_ids)) if flips[i] == 1],
                               mkt_size,
                               replace=False)

            replenishments = [draws[i] for i in range(draws.shape[0]) if flips[i] == 1]
            replaced = [helper.mkt_ids[i] for i in range(draws.shape[0]) if flips[i] == 1]
            swapped_pairs = zip(replaced, replenishments)
            remaining_vids = remaining_vids.union(replaced).difference(replenishments)

            # ensure that the remaining videos are distinct size is constant
            if len(list(remaining_vids)) != len(products) - mkt_size:
                print('remaining_vids', len(list(remaining_vids)))
                print('curr_vids', helper.mkt_ids)
                print('replenishments', replenishments)
                print('replaced', replaced)
                print('flips', flips)
                print('draws', draws)
                assert False

            for old, new in swapped_pairs:
                helper.replace_item(old, new)

            if t >= snapshot_start and rng.binomial(1, snapshotting_prob) > 0:
                snapshot_dict[snapshot_num] = (copy.deepcopy(helper.mkt_ids), copy.deepcopy(helper.market))
                snapshot_num += 1

        for prod in helper.mkt_ids:
            mkt_idx = helper.mkt_ids.index(prod)
            helper.universe[prod].append(helper.market[mkt_idx])

        # rename everything by id name if given
        if id_name:
            for k in list(helper.universe.keys())[:]:
                helper.universe[(k, id_name)] = helper.universe.pop(k)

            for k in list(snapshot_dict.keys())[:]:
                snapshot_dict[(k, id_name)] = snapshot_dict.pop(k)

        return helper.universe, snapshot_dict, np.array(market_history)
