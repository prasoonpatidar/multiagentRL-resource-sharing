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
from configs.train_configs import train_config, get_train_config
from evaluation.get_agent import get_agent

# step 1: initialize one each type of RL agent --target agent
# step 2: random generate other types of RL agent (N-1) number of agents--as environment
# step 3: keep the environment the same, come the reward and utilities learned by the target agent

agents_list = list(train_config.keys())
print('RL algorithms included in the comparison ',agents_list)

def get_env_agents(seller_info, agents_list):
    num_env_agents= seller_info.count-1
    index_agents = random.sample(range(0, len(agents_list)), num_env_agents)
    env_agents = []
    name_env_agents = [agents_list[index] for index in index_agents]
    # a container to store policies
    train_configs = []
    for agent_id in index_agents:
        # agent order helps ensure matching the seller id with policy id
        train_config = get_train_config(agents_list[agent_id])
        agent = get_agent(train_config)

        env_agents.append(agent)
        train_configs.append(train_config)
    return env_agents, name_env_agents, train_configs

def get_target_agents():
    target_agents = []
    train_configs = []
    for agent_id in range(0,len(agents_list)):
        train_config = get_train_config(agents_list[agent_id])
        seller_agent = get_agent(train_config)

        target_agents.append(seller_agent )
        train_configs.append(train_config)
    return target_agents, train_configs

def initialize_agents(agents, seller_info, agents_list,buyer_info, logger, market_name, train_configs, comp_agentsName):
    sellers = []
    policies = []

    for id in range(0, len(agents)):
        agent = agents[id]
        train_config = train_configs[id]
        # Load a policy
        results_file = f'../results/training/{market_name}_{comp_agentsName[id]}.pb'
        if os.path.exists(results_file):
            results_dir = pickle.load(open(results_file, 'rb'))
        else:
            logger.error("policy file not present, exiting")
            exit(1)
        policy = results_dir['seller_policies'][id]

        seller, logger = agent.initialize_agent(seller_info, buyer_info, train_config,
                      logger, compare=True, agentNum=1,
                                        is_trainer=False, results_dir=results_dir, seller_id=id)
        sellers.append(seller)
        policies.append(policy)
    return sellers, policies

def compare_agents(seller_info):
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
        sellers.append(target_agents[seller_id])

        agents.append( agents_list[seller_id])
        configs.append( target_configs[seller_id])

        # append comparision plans to list
        agents_comp.append(sellers)
        agent_names.append(agents)
        train_configs.append(configs)
    return agents_comp,  agent_names, train_configs

def iterate_policy(y, yAll, seller,helper, buyer_info, all_seller_actions, train_iter,seller_info, logger, train_config ):
    # calculate the choose probability
    prob = helper.choose_prob(y, compare=True, yAll=yAll)

    # Save prices in history
    price = 1 / y

    cumulativeBuyerExperience = helper.cumlativeBuyerExp(buyer_info, seller)
    X = helper.getPurchases(buyer_info, cumulativeBuyerExperience, y, prob)

    # Save purchased history
    purchase = X.sum(axis=0)


    # Get Buyer utilities and penalties in history
    buyerUtility = helper.buyerUtilitiesCalculator(X, y, buyer_info.V, buyer_info.a_val, prob,
                                              buyer_info.count,
                                              cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)

    buyerPenalty = helper.buyerPenaltiesCalculator(X, y, buyer_info.V, buyer_info.a_val, buyer_info.count,
                                              cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)

    seller_utility, seller_penalty, seller_provided_resource = \
        helper.evaluation(seller, train_config, all_seller_actions, yAll, X,
                          train_iter, seller_info, logger, train=False)

    # Get seller utilties and penalties in history
    seller_utility = np.array(seller_utility)
    seller_penalty = np.array(seller_penalty)

    # update provided resources history
    seller_provided_resource = np.array(seller_provided_resource)

    return price, purchase, buyerUtility, buyerPenalty, seller_utility, seller_penalty, seller_provided_resource

def logger_handle(logger_pass):
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('Compare policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    return logger

def compare_policy(seller_info, buyer_info, train_configs, sellers,agents, logger, policies, comp_agentsName, iterations ):

    # Get Containers to record history(Interesting insight: append in python list is O(1))
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []

    # Start Loop for training
    logger.info("Starting compare iterations...")
    start_time = time.time()
    for train_iter in range(0, iterations):

        if train_iter % 1000 == 0:
            logger.info("Finished %d compare iterations in %.3f secs..." % (train_iter, time.time() - start_time))

        # get the prices for all seller agents
        yAll = []
        all_seller_actions = []
        for s_id in range(0, len(sellers)):
            helper = agents[s_id]
            seller = sellers[s_id]
            train_config = train_configs[s_id]
            y, actions, action_number = helper.get_ys(seller, train_config, seller_info)
            yAll.append(y)
            all_seller_actions.append(actions)
        # change to 2D list to 1D list
        list(itertools.chain.from_iterable(yAll))
        list(itertools.chain.from_iterable(all_seller_actions))

        # run through sellers to update policy

        # a container to store each agent's information in one iteration
        prices = []
        purchases = []
        buyerUtilities = []
        buyerPenalties = []
        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []
        for seller_id in range(0,len(sellers)):
            helper = agents[seller_id]
            seller = sellers[seller_id]
            train_config = train_configs[seller_id]
            actions = all_seller_actions[seller_id]
            # evaluate each seller separately
            price, purchase, \
            buyerUtility, buyerPenalty, \
            seller_utility, seller_penalty, \
            seller_provided_resource = iterate_policy(yAll[seller_id], yAll, seller,helper, buyer_info,
                                                      actions, train_iter,seller_info, logger, train_config )
            prices.append(price)
            purchases.append(purchase)

            buyerUtilities.append(buyerUtility)
            buyerPenalties.append(buyerPenalty)

            seller_utilities.append(seller_utility)
            seller_penalties.append(seller_penalty)
            seller_provided_resources.append(seller_provided_resource)

        # Save purchased history
        # list(itertools.chain.from_iterable(prices))
        price_history.append(prices)
        # list(itertools.chain.from_iterable(purchases))
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        # list(itertools.chain.from_iterable(buyerUtilities))
        buyer_utility_history.append(buyerUtilities)
        # list(itertools.chain.from_iterable(buyerPenalties))
        buyer_penalty_history.append(buyerPenalties)

        # Get seller utilties and penalties in history
        # list(itertools.chain.from_iterable(seller_utilities))
        # list(itertools.chain.from_iterable(seller_penalties))
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)

        # update provided resources history
        # list(itertools.chain.from_iterable(seller_provided_resources))
        provided_resource_history.append(seller_provided_resources)

    # Create final results dictionary
    compare_dict = {
       'compared_agents':comp_agentsName,
        'seller_policies': policies,
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

def run_comparison(seller_info, buyer_info, logger_pass, market_name, iterations):
    # Initialize the logger
    logger = logger_handle(logger_pass)
    agents_comp, agent_names, train_configs = compare_agents(seller_info)
    # a container to store the comparison results
    compare_results = []
    for s_id in range(0,len(agents_comp)):
        # get the name list of the compared agents, the last agent is the target agent
        comp_agentsName = agent_names[s_id]

        # initialize the comparison parameters
        train_config = train_configs[s_id]

        agents = agents_comp[s_id]
        sellers, policies = initialize_agents(agents, seller_info, agents_list,buyer_info, logger, market_name, train_config, comp_agentsName)
        # call the comparison function
        compare_dict = compare_policy(seller_info, buyer_info, train_config, sellers,agents, logger, policies, comp_agentsName, iterations)
        compare_results.append(compare_dict)
    return compare_results

