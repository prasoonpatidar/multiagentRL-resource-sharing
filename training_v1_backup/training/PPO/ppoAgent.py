'''
Class to handle PPO Seller Agents
'''

import numpy as np


from training.PPO.network import DefaultNN



class ppoAgent:
    def __init__(self, seller_id, max_resources, cost_per_unit, action_number,
                 aux_price_min, aux_price_max, seller_idle_penalty, seller_count,
                 buyer_count, seller_policy=None, is_trainer=True):

        # get basic information
        self.cost_per_unit = cost_per_unit
        self.id = seller_id
        self.max_resources = max_resources
        self.action_number = action_number
        self.penalty_coeff = seller_idle_penalty
        self.seller_count = seller_count
        self.buyer_count = buyer_count
        self.y_min = aux_price_min
        self.y_max = aux_price_max



        # load policy and value networks
        # in dim: last round prices, out dim, total available actions
        self.actor = DefaultNN(seller_count, action_number)
        self.critic = DefaultNN(seller_count, 1)

        # containers for bookkeeping
        self.__providedResources = [np.zeros(self.buyer_count)]
        self.__demandedResources = [np.zeros(self.buyer_count)]


