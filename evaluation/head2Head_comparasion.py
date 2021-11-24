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

def get_env_agents(seller_info, agents_list,buyer_info, logger, market_config):
    num_env_agents= seller_info.count-1
    env_agents = random.sample(range(0, len(agents_list)), num_env_agents)
    env_sellers = []
    agent_order = -1
    name_env_agents = agents_list[env_agents]
    # a container to store policies
    policies = []
    for agent_id in env_agents:
        # agent order helps ensure matching the seller id with policy id
        agent_order += 1
        train_config = get_train_config(agents_list[agent_id])
        agent = get_agent(train_config)

        # Load a policy
        results_file = f'results/training/{market_config}_{train_config}.pb'
        if os.path.exists(results_file):
            results_dir = pickle.load(open(results_file, 'rb'))
        else:
            logger.error("policy file not present, exiting")
            exit(1)

        policy = results_dir['seller_policies'][agent_order]
        seller =agent.initialize_agent( seller_info, buyer_info, train_config,
                      logger, compare=True, agentNum=1,
                       is_trainer=False, results_dir=results_dir, seller_id=agent_order)
        env_sellers.append(seller)
        policies.append(policy)
    return env_sellers, policies,  name_env_agents

def get_target_agents(seller_info, buyer_info, logger, market_config):
    target_sellers = []
    agent_order = seller_info.count
    policies = []
    for agent in agents_list:
        train_config = get_train_config(agent)
        seller_agent = get_agent(train_config)

        # Load a policy
        results_file = f'results/training/{market_config}_{train_config}.pb'
        if os.path.exists(results_file):
            results_dir = pickle.load(open(results_file, 'rb'))
        else:
            logger.error("policy file not present, exiting")
            exit(1)
        policy = results_dir['seller_policies'][agent_order]
        seller = seller_agent.initialize_agent(seller_info, buyer_info, train_config,
                                        logger, compare=True, agentNum=1,
                                        is_trainer=False, results_dir=results_dir, seller_id=agent_order)
        target_sellers.append(seller)
        policies.append(policy)
    return target_sellers, policies

def compare_agents(seller_info, buyer_info, logger, market_config):
    env_sellers, env_policies,  name_env_agents = get_env_agents(seller_info, agents_list,buyer_info, logger, market_config)
    target_sellers, target_policies = get_target_agents(seller_info, buyer_info, logger, market_config)
    sellers_comp = []
    policies = []
    agent_names = []
    for seller_id in range(0, len(target_sellers)):
        # add one target seller to the environment
        sellers_comp.append(list(env_sellers)+target_sellers[seller_id])
        policies.append(list(env_policies)+target_policies[seller_id])
        agent_names.append(list(name_env_agents) + agents_list[seller_id])
    return sellers_comp, policies, agent_names

def eval_policy(y, yAll, seller, buyer_info):
    prob, y = seller.choose_prob(y, compare=True, yAll=yAll)

    # Save prices in history
    prices = 1 / y

    cumulativeBuyerExperience = seller.cumlativeBuyerExp(buyer_info, seller)
    X = seller.getPurchases(buyer_info, cumulativeBuyerExperience, y, prob)

    # Save purchased history
    purchases = X.sum(axis=0)


    # Get Buyer utilities and penalties in history
    buyerUtilities = seller.buyerUtilitiesCalculator(X, y, buyer_info.V, buyer_info.a_val, prob,
                                              buyer_info.count,
                                              cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)

    buyerPenalties = seller.buyerPenaltiesCalculator(X, y, buyer_info.V, buyer_info.a_val, buyer_info.count,
                                              cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)

    # Run through sellers to update policy
    seller_utilities = []
    seller_penalties = []
    seller_provided_resources = []

    for j in range(0, len(seller)):
        x_j = X[j]
        tmpSellerUtility, tmpSellerPenalty, z_j = seller[j].updateQ(x_j, 0., 0., yAll)
        seller_utilities.append(tmpSellerUtility)
        seller_penalties.append(tmpSellerPenalty)
        seller_provided_resources.append(z_j)

    # Get seller utilties and penalties in history
    seller_utilities = np.array(seller_utilities)
    seller_penalties = np.array(seller_penalties)

    # update provided resources history
    seller_provided_resources = np.array(seller_provided_resources)

    return prices, purchases, buyerUtilities, buyerPenalties, seller_utilities, seller_penalties, seller_provided_resources

def logger_handle(logger_pass):
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('Q_learn_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    return logger

def compare_policy(seller_info, buyer_info, train_config, sellers, logger, policies, compared_agents ):
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
    for train_iter in range(0, train_config.iterations):

        if train_iter % 1000 == 0:
            logger.info("Finished %d compare iterations in %.3f secs..." % (train_iter, time.time() - start_time))

        # get the prices for all seller agents
        yAll = []
        for seller in sellers:
            y = seller.get_ys([seller], train_config, seller_info)
            yAll.append(y)

        # a container to store each agent's information in one iteration
        prices = []
        purchases = []
        buyerUtilities = []
        buyerPenalties = []
        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []
        for seller in sellers:
            # evaluate each seller separately
            price, purchase, buyerUtility, buyerPenalty,  \
            seller_utility, seller_penalty, seller_provided_resource \
                = eval_policy(y, yAll, [seller], buyer_info)

            prices.append(price)
            purchases.append(purchase)
            buyerUtilities.append(buyerUtility)
            buyerPenalties.append(buyerPenalty)
            seller_utilities.append(seller_utility)
            seller_penalties.append(seller_penalty)
            seller_provided_resources.append(seller_provided_resource)

        # Save purchased history
        list(itertools.chain.from_iterable(prices))
        price_history.append(prices)
        list(itertools.chain.from_iterable(purchases))
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        list(itertools.chain.from_iterable(buyerUtilities))
        buyer_utility_history.append(buyerUtilities)
        list(itertools.chain.from_iterable(buyerPenalties))
        buyer_penalty_history.append(buyerPenalties)

        # Get seller utilties and penalties in history
        list(itertools.chain.from_iterable(seller_utilities))
        list(itertools.chain.from_iterable(seller_penalties))
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)

        # update provided resources history
        list(itertools.chain.from_iterable(seller_provided_resources))
        provided_resource_history.append(seller_provided_resources)

    # Create final results dictionary
    compare_dict = {
       'compared_agents':compared_agents,
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

def run_comparison(seller_info, buyer_info, logger_pass, market_config):
    # Initialize the logger
    logger = logger_handle(logger_pass)
    sellers_comp, policies, agent_names = compare_agents(seller_info, buyer_info, logger, market_config)
    # a container to store the comparison results
    compare_results = []
    for sellers_id in range(0,len(sellers_comp)):
        compared_agents = agent_names[sellers_id]
        compare_dict = compare_policy(seller_info, buyer_info, train_config, sellers_comp[sellers_id], logger, policies, compared_agents )

        compare_results.append(compare_dict)
    return compare_results

