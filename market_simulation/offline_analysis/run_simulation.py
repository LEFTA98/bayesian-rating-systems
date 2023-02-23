"""
class for actually running the simulations.
"""

from typing import List, Union
import copy

import pandas as pd
import numpy as np

from .product_utils import ProductHelper, sample_chosen_df
from .choice_models.multi_armed_bandit import random_argmax, ts_action

DEFAULT_SELECTION = "thompson_sampling"
DEFAULT_NUM_TIMESTEPS = 5000000
DEFAULT_RHO = 0.01
DEFAULT_MKT_SIZE = 5
DEFAULT_NUM_USERS = 1
DEFAULT_SNAPSHOT_PROB = 0.001

DEFAULT_CHOICE_FN_MAPPING = {"thompson_sampling": ts_action}


class SimRunner:

    def __init__(self, ratings_style: str, selection_style: str = DEFAULT_SELECTION):
        self.ratings_style = ratings_style
        self.selection_style = selection_style

    def run_single_simulation(self,
                              input_df: pd.DataFrame,
                              videos: List[str],
                              priors: np.ndarray,
                              timesteps: int = DEFAULT_NUM_TIMESTEPS,
                              rho: float = DEFAULT_RHO,
                              mkt_size: int = DEFAULT_MKT_SIZE,
                              num_users: int = DEFAULT_NUM_USERS,
                              snapshot_start: Union[bool, int] = None,
                              snapshotting_prob: int = DEFAULT_SNAPSHOT_PROB,
                              seed: Union[bool, int] = None,
                              id_name: Union[bool, str] = None):
        rng = np.random.default_rng(seed)
        sampling_action = DEFAULT_CHOICE_FN_MAPPING[self.selection_style]
        if not snapshot_start:
            snapshot_start = timesteps // 5
        product_data = dict(zip(videos, [[] for _ in range(len(videos))]))
        priors_list = [copy.deepcopy(priors) for _ in range(len(videos))]
        priors_dict = dict(zip(videos, priors_list))
        snapshot_dict = dict()
        snapshot_num = 1

        curr_vids = np.array(list(rng.choice(videos, mkt_size, replace=False)))
        remaining_vids = set(videos).difference(set(curr_vids))

        helper = ProductHelper(product_data, curr_vids, list(curr_vids), priors_dict)
        market_history = []

        for t in range(timesteps):
            if (t + 1) % (timesteps // 10) == 0:
                print(f'{t + 1}/{timesteps}')

            market_history.append(copy.deepcopy(helper.mkt_ids))
            latest_sims = np.array([item[-1] for item in helper.market])
            successes, failures = latest_sims[:, 0], latest_sims[:, 1]
            actions = range(mkt_size)
            for m in range(num_users):
                a = sampling_action(actions, successes, failures, rng=None)
                chosen_action_global_index = videos.index(helper.mkt_ids[a])
                market_history[-1].append(copy.deepcopy(helper.mkt_ids[a]))
                like = sample_chosen_df(videos, input_df, chosen_action_global_index, rng=None)

                # update prior
                helper.pull_arm_update_market(a, like)

            # replenish the indices
            flips = rng.binomial(1, rho, mkt_size)
            draws = rng.choice(list(remaining_vids) +
                               [helper.mkt_ids[i] for i in range(len(helper.mkt_ids)) if flips[i]==1], mkt_size,replace=False)

            replenishments = flips * draws
            replaced = flips * helper.mkt_ids
            swapped_pairs = zip(list(replaced[replaced != 0].flatten()), list(replenishments[replenishments != 0].flatten()))
            replenishments = set(replenishments[replenishments != 0].flatten().astype(int))
            replaced = set(replaced[replaced != 0].flatten().astype(int))
            remaining_vids = remaining_vids.union(replaced).difference(replenishments)

            # ensure that the remaining videos are distinct size is constant
            if len(list(remaining_vids)) != len(videos) - mkt_size:
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
