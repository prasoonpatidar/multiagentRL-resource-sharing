'''
Main wrapper for qlearning training
'''
import random
import pickle
import numpy as np

from training.QLearning.qAgent import qAgent
from training.seller_utils import action2y, ydiff2action


def initialize_agents(seller_info, buyer_info, train_config, logger, evaluate=False):
    # get required parameters for Q learning algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_config.action_count

    sellers = []
    for seller_id in range(seller_info.count):
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
                           aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                           buyer_info.count, train_config, evaluate)
        sellers.append(tmpSeller)
    logger.info(f"Initialized {seller_info.count} seller agents for training")

    return sellers


def initialize_single_eval_agent(seller_id, seller_info, buyer_info, train_config, logger):
    # get required parameters for Q learning algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_config.action_count
    max_resources = seller_info.max_resources[seller_id]
    cost_per_unit = seller_info.per_unit_cost[seller_id]
    tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
                       aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                       buyer_info.count, train_config, evaluate=True)
    return tmpSeller


def get_initial_state(seller_info, buyer_info, train_config, logger):
    init_state = np.random.randint(0, train_config.action_count, seller_info.count)
    return init_state


def get_actions(sellers, state):
    actions = []
    ys = []
    for j in range(len(sellers)):
        seller_action = sellers[j].get_next_action()
        actions.append(seller_action)
        ys.append(action2y(seller_action, sellers[j].action_number, sellers[j].y_min, sellers[j].y_max))

    return np.array(actions), np.array(ys)


def get_next_state(sellers, state, actions):
    # for q learning next state is actions taken in this state
    return actions

def get_next_state_from_ys(ys, action_number, y_min,y_max, seller_count):
    ydiff = ys - y_min
    actions = ydiff2action(ydiff,action_number, y_min,y_max)
    return actions


def step(sellers, train_iter, env_state, actions, next_state, X, seller_utilities, seller_penalties,
         distributed_resources, train_config, evaluate=False):
    for j in range(len(sellers)):
        if not evaluate:
            seller_reward = seller_utilities[j] + seller_penalties[j]
            sellers[j].learning_rate = 1 / (20 + train_iter)
            sellers[j].updateQ(seller_reward, sellers[j].learning_rate, sellers[j].discount_factor)
            sellers[j].updatePolicy(sellers[j].explore_prob)
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
