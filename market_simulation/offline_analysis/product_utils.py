"""Helper class and functions for running simulations that keeps track of the current market, ids, and priors."""

import numpy as np
import copy


def sample_chosen_df(videos, chosen_df, action_index, rng=None):
    """helper function for sampling products from an input dataset."""
    if not rng:
        rng = np.random.default_rng()

    vid = videos[action_index]
    seen_like = chosen_df[chosen_df['video_id'] == vid].sample(1, random_state=rng).iloc[0]['liked']
    return seen_like


class ProductHelper:
    """Class for helping keep track of all the products we have."""

    def __init__(self, universe, starting_market, mkt_ids, priors):
        self.universe = universe
        self.mkt_ids = mkt_ids
        self.priors = priors
        self.market = [np.array([copy.deepcopy(self.priors)[k]]) for k in starting_market]

    def pull_arm(self, action, like):
        arr = self.market[action]
        latest_action = copy.deepcopy(arr[-1])
        latest_action[like] += 1
        self.market[action] = np.append(arr, [latest_action], axis=0)

    def pull_arm_update_market(self, action, like):
        self.pull_arm(action, like)
        for i in range(len(self.market)):
            if not np.array_equal(i, action):
                self.market[i] = np.append(self.market[i], [self.market[i][-1]], axis=0)

    def replace_item(self, old_id, new_id):
        old_idx = self.mkt_ids.index(old_id)
        self.universe[old_id].append(copy.deepcopy(self.market[old_idx]))
        self.mkt_ids[old_idx] = new_id
        self.market[old_idx] = np.array([copy.deepcopy(self.priors[new_id])])