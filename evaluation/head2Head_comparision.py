import numpy as np
import matplotlib.pyplot as plt
import time
import math
import logging
import random
import itertools
import os, sys
import pickle

# import custom libraries
from configs.train_configs import train_configs, get_train_config
from training.get_trainer import get_trainer
import training.seller_utils as seller_utils
import training.buyer_utils as buyer_utils

# step 1: initialize one each type of RL agent --target agent
# step 2: random generate other types of RL agent (N-1) number of agents--as environment
# step 3: keep the environment the same, come the reward and utilities learned by the target agent

agents_list = ['q_r1','wolf_r1','dqn_r2','ddqn_r2','dqn_duel_r2']
print('RL algorithms included in the comparison ',agents_list)

def get_env_agents(seller_info, agents_list):
    num_env_agents= seller_info.count
    # index_agents = random.sample(range(0, len(agents_list)), num_env_agents)
    index_agents = np.random.choice(range(0, len(agents_list)), num_env_agents)
    env_agents = []
    name_env_agents = [agents_list[index] for index in index_agents]
    # a container to store policies
    train_configs = []
    for agent_id in index_agents:
        # agent order helps ensure matching the seller id with policy id
        train_config = get_train_config(agents_list[agent_id])
        agent = get_trainer(train_config.rl_trainer)

        env_agents.append(agent)
        train_configs.append(train_config)
    return env_agents, name_env_agents, train_configs

def get_target_agents():
    target_agents = []
    train_configs = []
    for agent_id in range(0,len(agents_list)):
        train_config = get_train_config(agents_list[agent_id])
        seller_agent = get_trainer(train_config.rl_trainer)

        target_agents.append(seller_agent )
        train_configs.append(train_config)
    return target_agents, train_configs

def init_agents(agents, seller_info, agents_list,buyer_info, logger, market_name, train_configs, comp_agentsName):
    sellers = []
    for id in range(0, len(agents)):
        agent = agents[id]
        train_config = train_configs[id]
        # seller = agent.initialize_agents()
        seller= agent.initialize_single_eval_agent(id, seller_info, buyer_info, train_config, logger)
        sellers.append(seller)
    return sellers

def compare_agents(compare_seller_id, seller_info):
    env_agents, name_env_agents, env_configs = get_env_agents(seller_info, agents_list)
    target_agents, target_configs = get_target_agents()
    agents_comp = []
    agent_names = []
    train_configs = []
    for seller_id in range(0, len(target_agents)):
        sellers = env_agents.copy()
        agents = name_env_agents.copy()
        configs = env_configs.copy()
        # add one target seller to the environment

        sellers[compare_seller_id]=target_agents[seller_id]

        agents[compare_seller_id]=agents_list[seller_id]
        configs[compare_seller_id]=target_configs[seller_id]

        # append comparision plans to list
        agents_comp.append(sellers)
        agent_names.append(agents)
        train_configs.append(configs)
    return agents_comp,  agent_names, train_configs


def logger_handle(logger_pass):
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('Compare policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    return logger

def compare_policy(seller_info, buyer_info, train_configs, sellers,agents, logger, comp_agentsName, iterations ):

    # Get Containers to record history(Interesting insight: append in python list is O(1))
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []

    # get env_states and next_states for all agents
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_configs[0].action_count
    ys_init = np.random.uniform(low=aux_price_min, high=aux_price_max, size=seller_info.count)
    env_states = []
    for s_id in range(0, len(sellers)):
        env_states.append(agents[s_id].get_next_state_from_ys(ys_init, action_number, aux_price_min,aux_price_max,
                                                              seller_info.count))
    next_states = [None] * seller_info.count

    # Start Loop for training
    logger.info("Starting compare iterations...")
    start_time = time.time()
    for compare_iter in range(0, iterations):

        if compare_iter % 100 == 0:
            logger.info("Finished %d compare iterations in %.3f secs..." % (compare_iter, time.time() - start_time))

        # get the prices for all seller agents
        yAll = []
        all_seller_actions = []
        for s_id in range(0, len(sellers)):
            helper = agents[s_id]
            seller = sellers[s_id]
            train_config = train_configs[s_id]
            action, ys =  helper.get_actions([seller], env_states[s_id])
            yAll.append(ys[0])
            all_seller_actions.append(action[0])

        probAll, yAll = seller_utils.choose_prob(yAll, compare=False, yAll=None)

        # Save prices in history
        prices = 1 / yAll
        price_history.append(prices)

        # get buyer actions based on their experiences
        cumulativeBuyerExperience = buyer_utils.getBuyerExperience(sellers, buyer_info)
        X = buyer_utils.getPurchases(buyer_info, cumulativeBuyerExperience, yAll, probAll)

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # get buyer utilities and penalties
        buyer_utilities, buyer_penalties = buyer_utils.get_buyer_rewards(X, yAll, probAll, cumulativeBuyerExperience, buyer_info)
        buyer_utility_history.append(buyer_utilities)
        buyer_penalty_history.append(buyer_penalties)

        # get seller utilities based on current state and actions
        seller_utilities, seller_penalties, distributed_resources = seller_utils.get_rewards(sellers, X, yAll,probAll)
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)
        provided_resource_history.append(distributed_resources)

        # get next states based on current state and actions, and take eval step
        for s_id in range(0, len(sellers)):
            env_states[s_id] = agents[s_id].get_next_state_from_ys(yAll, action_number, aux_price_min, aux_price_max,
                                                                  seller_info.count)
            sellers[s_id].add_purchase_history(X[s_id],distributed_resources[s_id])


    # Create final results dictionary
    compare_dict = {
       'compared_agents':comp_agentsName,
        'buyer_info': buyer_info,
        'seller_info': seller_info,
        'price_history': price_history,
        'seller_utilties': seller_utility_history,
        'seller_penalties': seller_penalty_history,
        'buyer_utilties': buyer_utility_history,
        'buyer_penalties': buyer_penalty_history,
        'demand_history': purchase_history,
        'supply_history': provided_resource_history
    }

    return compare_dict

def run_comparison(compare_seller_id, seller_info, buyer_info, logger_pass, market_name, iterations):
    # Initialize the logger
    logger = logger_handle(logger_pass)
    agents_comp, agent_names, train_configs = compare_agents(compare_seller_id,seller_info)

    # add action counts in all train configs
    for config_list in train_configs:
        for config in config_list:
            config.action_count = seller_info.action_count
            policy_dir_path = config.policy_store.split("/")
            config.policy_store = '/'.join([policy_dir_path[0],market_name,config.config_name])

    # a container to store the comparison results
    compare_results = []
    for s_id in range(0,len(agents_comp)):
        # get the name list of the compared agents, the last agent is the target agent
        comp_agentsName = agent_names[s_id]

        # initialize the comparison parameters
        train_config = train_configs[s_id]

        agents = agents_comp[s_id]
        sellers = init_agents(agents, seller_info, agents_list,buyer_info, logger, market_name, train_config, comp_agentsName)
        # call the comparison function
        compare_dict = compare_policy(seller_info, buyer_info, train_config, sellers,agents, logger, comp_agentsName, iterations)
        compare_dict.update({'compare_seller_id':compare_seller_id})
        compare_results.append(compare_dict)
    return compare_results

