'''
Main wrapper for SAC training
'''
import numpy as np

from training.SAC.sacAgent import sacAgent
from training.seller_utils import ydiff2action


def initialize_agents(seller_info, buyer_info, train_config, logger, evaluate=False):
    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price

    # initialize seller agents
    sellers = []
    action_number = train_config.action_count

    for seller_id in range(seller_info.count):
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = sacAgent(seller_id, max_resources, cost_per_unit, action_number,
                             aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                             buyer_info.count, train_config, evaluate)
        if evaluate:
            tmpSeller.load_model()
        sellers.append(tmpSeller)

    logger.info(f"Initialized {seller_info.count} seller agents for training")
    return sellers


def initialize_single_eval_agent(seller_id, seller_info, buyer_info, train_config, logger):
    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price

    # initialize seller agent
    max_resources = seller_info.max_resources[seller_id]
    cost_per_unit = seller_info.per_unit_cost[seller_id]
    action_number = train_config.action_count
    tmpSeller = sacAgent(seller_id, max_resources, cost_per_unit, action_number,
                         aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                         buyer_info.count, train_config, True)
    tmpSeller.load_model()
    return tmpSeller


def get_initial_state(seller_info, buyer_info, train_config, logger):
    init_state = np.random.randint(0, train_config.action_count, seller_info.count)
    return init_state


def get_actions(sellers, state):
    ydiffActions = []
    for j in range(len(sellers)):
        ydiffAction = sellers[j].policy_net.get_action(state, deterministic=sellers[j].deterministic)
        if not np.isnan(ydiffAction):
            ydiffActions.append(ydiffAction)
        else:
            ydiffActions.append(0.)
    ydiffActions = np.array(ydiffActions).flatten()
    ys = sellers[j].y_min + ydiffActions
    return ydiffActions, np.array(ys)


def get_next_state(sellers, state, actions):
    return ydiff2action(actions, sellers[0].action_number, sellers[0].y_min, sellers[0].y_max)

def get_next_state_from_ys(ys, action_number, y_min,y_max, seller_count):
    ydiff = ys - y_min
    actions = ydiff2action(ydiff,action_number, y_min,y_max)
    return actions

def step(sellers, train_iter, env_state, actions, next_state, X, seller_utilities, seller_penalties,
         distributed_resources, train_config, evaluate=False):
    for j in range(len(sellers)):
        if not evaluate:
            seller_reward = seller_utilities[j] + seller_penalties[j]
            sellers[j].replay_buffer.push(env_state, [actions[sellers[j].id]], seller_reward, next_state, False)
            if len(sellers[j].replay_buffer) > train_config.batch_size:
                for i in range(train_config.update_itr):
                    _ = sellers[j].update(train_config.batch_size, reward_scale=10.,
                                          auto_entropy=train_config.auto_entropy,
                                          target_entropy=-1. * sellers[j].action_size)

            if train_iter % (train_config.update_step_size) == 0:
                sellers[j].save_model()

        # add current state history
        sellers[j].add_purchase_history(X[j], distributed_resources[j])

    return None


def get_policies(sellers):
    ...


def save_policies(sellers):
    ...
