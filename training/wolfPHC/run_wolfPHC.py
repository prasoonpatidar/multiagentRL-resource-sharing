'''
This is the main RL Algorithm file for WolfPHC. it contains wrapper functions to learn and evaluate RL policies.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import logging

# custom project libraries
from training.wolfPHC.wolfphcAgent import wolfphcAgent
from training.wolfPHC.utils import sellerAction2y,buyerPurchaseCalculator
from training.wolfPHC.utils import buyerUtilitiesCalculator, buyerPenaltiesCalculator
'''
Main learner function to run policy for sellers using WoLF-PHC agent
'''


def learn_policy(run_config, seller_info, buyer_info, train_config, logger_pass):
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('WoLF_learn_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # get required parameters for WolFPHC algorithm
    discount_factor = train_config.discount_factor
    learning_rate = train_config.learning_rate
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers = []
    action_number = train_config.action_count
    for seller_id in range(seller_info.count):
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = wolfphcAgent(seller_id, max_resources, cost_per_unit, action_number,
                                 aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                                 buyer_info.count)
        sellers.append(tmpSeller)
    logger.info(f"Initialized {seller_info.count} seller agents")

    # Get Containers to record history(Interesting insight: append in python list is O(1))
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []

    # Start Loop for training
    logger.info("Starting training iterations...")
    start_time = time.time()
    for train_iter in range(0, train_config.iterations):

        if train_iter % 1000 == 0:
            logger.info("Finished %d training iterations in %.3f secs..." % (train_iter, time.time() - start_time))

        # loop parameters
        lr_win =  1 / (500 + 0.1 * train_iter)

        # get actions from all sellers
        all_seller_actions = []
        for tmpSeller in sellers:
            all_seller_actions.append(tmpSeller.get_next_action())
        all_seller_actions = np.array(all_seller_actions)
        yAll = sellerAction2y(all_seller_actions, action_number, aux_price_min, aux_price_max)


        # Save prices in history
        prices = 1 / yAll
        price_history.append(prices)

        # get the buyer experience with sellers based on previous purchases
        cumulativeBuyerExperience = np.zeros((buyer_info.count, seller_info.count))
        for i in range(0,buyer_info.count):
            for j in range(0,seller_info.count):
                cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)

        # get the amount of resources purchased by each device based on y
        X = []
        for i in range(0,buyer_info.count):
            X_i = buyerPurchaseCalculator(cumulativeBuyerExperience[i,:], yAll,buyer_info.V[i],buyer_info.a_val[i]
                                          ,seller_info.count,buyer_info.unfinished_task_penalty)
            X.append(X_i)
        X = np.array(X).T

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X,yAll,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X,yAll,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_penalty_history.append(buyerPenalties)

        # Run through sellers to update policy

        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []

        for j in range(0, seller_info.count):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].updateQ(all_seller_actions, x_j, learning_rate,
                                                                         discount_factor, seller_info.count,
                                                                         action_number)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)
            sellers[j].updateMeanPolicy()  # 更新平均策略 update mean policy
            sellers[j].updatePolicy(lr_win)  # 更新策略 update policy
            sellers[j].updateState()  # 更新状态 update state

        # Get seller utilties and penalties in history
        seller_utilities = np.array(seller_utilities)
        seller_penalties = np.array(seller_penalties)
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)

        #update provided resources history
        seller_provided_resources = np.array(seller_provided_resources)
        provided_resource_history.append(seller_provided_resources)


    # Get final policies for sellers
    seller_policies = {}
    for j in range(seller_info.count):
        seller_policies[j] =sellers[j].get_policy()

    # Create final results dictionary
    results_dict = {
        'seller_policies':seller_policies,
        'buyer_info':buyer_info,
        'seller_info':seller_info,
        'price_history':price_history,
        'seller_utilties':seller_utility_history,
        'seller_penalties':seller_penalty_history,
        'buyer_utilties':buyer_utility_history,
        'buyer_penalties':buyer_penalty_history,
        'demand_history':purchase_history,
        'supply_history':provided_resource_history
    }

    return results_dict

def eval_policy(seller_info, buyer_info, train_config, results_dir, logger_pass):
    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('WoLF_evaluate_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # get required parameters for WolFPHC algorithm
    discount_factor = train_config.discount_factor
    learning_rate = train_config.learning_rate
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")
    #
    # # initialize seller agents
    sellers = []
    action_number = train_config.action_count
    for seller_id in range(seller_info.count):
        seller_policy = results_dir['seller_policies'][seller_id]
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = wolfphcAgent(seller_id, max_resources, cost_per_unit, action_number,
                                 aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                                 buyer_info.count, seller_policy, is_trainer=False)
        sellers.append(tmpSeller)
    logger.info(f"Initialized {seller_info.count} seller agents")

    # Get Containers to record history(Interesting insight: append in python list is O(1))
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []

    # Start Loop for evaluation
    logger.info("Starting evaluate iterations...")
    start_time = time.time()
    for eval_iter in range(0, train_config.iterations):

        if eval_iter % 100 == 0:
            logger.info("Finished %d evaluating iterations in %.3f secs..." % (eval_iter, time.time() - start_time))

        # get actions from all sellers
        all_seller_actions = []
        for tmpSeller in sellers:
            all_seller_actions.append(tmpSeller.get_next_action())
        all_seller_actions = np.array(all_seller_actions)
        yAll = sellerAction2y(all_seller_actions, action_number, aux_price_min, aux_price_max)


        # Save prices in history
        prices = 1 / yAll
        price_history.append(prices)

        # get the buyer experience with sellers based on previous purchases
        cumulativeBuyerExperience = np.zeros((buyer_info.count, seller_info.count))
        for i in range(0,buyer_info.count):
            for j in range(0,seller_info.count):
                cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)

        # get the amount of resources purchased by each device based on y
        X = []
        for i in range(0,buyer_info.count):
            X_i = buyerPurchaseCalculator(cumulativeBuyerExperience[i,:], yAll,buyer_info.V[i],buyer_info.a_val[i]
                                          ,seller_info.count,buyer_info.unfinished_task_penalty)
            X.append(X_i)
        X = np.array(X).T

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X,yAll,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X,yAll,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_penalty_history.append(buyerPenalties)

        # Run through sellers to update policy

        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []

        for j in range(0, seller_info.count):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].updateQ(all_seller_actions, x_j, learning_rate,
                                                                         discount_factor, seller_info.count,
                                                                         action_number)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)
            sellers[j].updateState()  # update state

        # Get seller utilties and penalties in history
        seller_utilities = np.array(seller_utilities)
        seller_penalties = np.array(seller_penalties)
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)

        #update provided resources history
        seller_provided_resources = np.array(seller_provided_resources)
        provided_resource_history.append(seller_provided_resources)

    # Create final results dictionary
    eval_dict = {
        'seller_policies':results_dir['seller_policies'],
        'buyer_info':buyer_info,
        'seller_info':seller_info,
        'price_history':price_history,
        'seller_utilties':seller_utility_history,
        'seller_penalties':seller_penalty_history,
        'buyer_utilties':buyer_utility_history,
        'buyer_penalties':buyer_penalty_history,
        'demand_history':purchase_history,
        'supply_history':provided_resource_history
    }

    return eval_dict