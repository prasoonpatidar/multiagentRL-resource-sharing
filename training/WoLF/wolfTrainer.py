'''
Main wrapper for WoLF-PHC training
'''

import numpy as np
import pickle

from training.WoLF.wolfAgent import wolfAgent
from training.WoLF.wolf_utils import allSellerActions2stateIndex
from training.seller_utils import action2y, ydiff2action


def initialize_agents(seller_info, buyer_info, train_config, logger, evaluate=False):
    # Initialize parameters

    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_config.action_count

    sellers = []
    for seller_id in range(seller_info.count):
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = wolfAgent(seller_id, max_resources, cost_per_unit, action_number,
                              aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                              buyer_info.count, train_config, evaluate)
        sellers.append(tmpSeller)

    logger.info(f"Initialized {seller_info.count} seller agents for training")

    return sellers


def initialize_single_eval_agent(seller_id, seller_info, buyer_info, train_config, logger):
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_config.action_count
    max_resources = seller_info.max_resources[seller_id]
    cost_per_unit = seller_info.per_unit_cost[seller_id]
    tmpSeller = wolfAgent(seller_id, max_resources, cost_per_unit, action_number,
                          aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                          buyer_info.count, train_config, evaluate=True)
    return tmpSeller


def get_initial_state(seller_info, buyer_info, train_config, logger):
    init_actions = np.random.randint(0, train_config.action_count, seller_info.count)
    init_state = allSellerActions2stateIndex(init_actions, seller_info.count, train_config.action_count)
    return init_state


def get_actions(sellers, state):
    actions = []
    ys = []
    for j in range(len(sellers)):
        seller_action = sellers[j].get_next_action(int(state))
        actions.append(seller_action)
        ys.append(action2y(seller_action, sellers[j].action_number, sellers[j].y_min, sellers[j].y_max))

    return np.array(actions), np.array(ys)


def get_next_state(sellers, state, actions):
    # next state is based on actions taken in this state
    next_state = allSellerActions2stateIndex(actions, sellers[0].seller_count, sellers[0].action_number)
    return next_state

def get_next_state_from_ys(ys, action_number, y_min, y_max, seller_count):
    ydiff = ys - y_min
    actions = ydiff2action(ydiff,action_number, y_min,y_max)
    next_state = allSellerActions2stateIndex(actions, seller_count, action_number)
    return next_state


def step(sellers, train_iter, env_state, actions, next_state, X, seller_utilities, seller_penalties,
         distributed_resources, train_config, evaluate=False):
    lr_win = 1 / (500 + 0.1 * train_iter)

    for j in range(len(sellers)):
        if not evaluate:
            seller_reward = seller_utilities[j] + seller_penalties[j]
            sellers[j].learning_rate = 1 / (20 + train_iter)
            sellers[j].updateQ(env_state, actions[j], seller_reward, sellers[j].learning_rate,
                               sellers[j].discount_factor)
            sellers[j].updateMeanPolicy(env_state)  # update mean policy
            sellers[j].updatePolicy(env_state, lr_win)  # update policy
        # add current state history
        sellers[j].add_purchase_history(X[j], distributed_resources[j])

    return None


def save_policies(sellers):
    for j in range(len(sellers)):
        pickle.dump(sellers[j].get_policy(), open(f'{sellers[j].policy_store}/{sellers[j].id}_policy.pb', 'wb'))
    return None


def get_policies(sellers):
    seller_policies = {}
    for j in range(len(sellers)):
        seller_policies[j] = sellers[j].get_policy()
    return seller_policies
