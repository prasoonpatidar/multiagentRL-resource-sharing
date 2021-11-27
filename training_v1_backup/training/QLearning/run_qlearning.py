'''
This is the main RL Algorithm file for QLearning. it contains wrapper functions to learn and evaluate RL policies.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import logging
from scipy.optimize import minimize, LinearConstraint

# custom libraries
from training.QLearning.qAgent import qAgent
from training.QLearning.utils import action2y, buyerPurchaseCalculator
from training.QLearning.utils import buyerPenaltiesCalculator, buyerUtilitiesCalculator


def learn_policy(run_config, seller_info, buyer_info, train_config, logger_pass):

    # Initialize the logger
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('Q_learn_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers = []
    action_number = train_config.action_count
    for seller_id in range(seller_info.count):
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
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
        lr =  1 / (20+train_iter)

        # get actions from all sellers
        actions = []
        for tmpSeller in sellers:
            actions.append(tmpSeller.get_next_action())
        actions = np.array(actions)
        ys = action2y(actions, action_number, aux_price_min, aux_price_max)

        # Save prices in history
        prices = 1 / ys
        price_history.append(prices)

        # get the buyer experience with sellers based on previous purchases
        cumulativeBuyerExperience = np.zeros((buyer_info.count, seller_info.count))
        for i in range(0,buyer_info.count):
            for j in range(0,seller_info.count):
                cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)

        # get the amount of resources purchased by each device based on y
        X = []
        for i in range(0,buyer_info.count):
            X_i = buyerPurchaseCalculator(cumulativeBuyerExperience[i,:], ys,buyer_info.V[i],buyer_info.a_val[i]
                                          ,seller_info.count,buyer_info.unfinished_task_penalty)
            X.append(X_i)
        X = np.array(X).T

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X,ys,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X,ys,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_penalty_history.append(buyerPenalties)

        # Run through sellers to update policy
        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []

        for j in range(0, seller_info.count):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].updateQ(actions, x_j, lr,
                                                                         train_config.discount_factor)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)
            sellers[j].updatePolicy(train_config.explore_prob)  # update policy

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
    logger_base = logger_pass.get('logger_base').getChild('Q_evaluate_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # get required parameters for Q-learning algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers = []
    action_number = train_config.action_count
    for seller_id in range(seller_info.count):
        seller_policy = results_dir['seller_policies'][seller_id]
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
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

    # Start Loop for training
    logger.info("Starting evaluate iterations...")
    start_time = time.time()
    for eval_iter in range(0, train_config.iterations):

        if eval_iter % 100 == 0:
            logger.info("Finished %d evaluating iterations in %.3f secs..." % (eval_iter, time.time() - start_time))

        # get actions from all sellers
        actions = []
        for tmpSeller in sellers:
            actions.append(tmpSeller.get_next_action())
        actions = np.array(actions)
        ys = action2y(actions, action_number, aux_price_min, aux_price_max)
        # ys [1, 2]
        # wolf-PHC [1]---1---
        # [1, 2, 3]
        # Save prices in history
        prices = 1 / ys
        price_history.append(prices)

        # get the buyer experience with sellers based on previous purchases
        cumulativeBuyerExperience = np.zeros((buyer_info.count, seller_info.count))
        for i in range(0,buyer_info.count):
            for j in range(0,seller_info.count):
                cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)

        # get the amount of resources purchased by each device based on y
        X = []
        for i in range(0,buyer_info.count):
            X_i = buyerPurchaseCalculator(cumulativeBuyerExperience[i,:], ys,buyer_info.V[i],buyer_info.a_val[i]
                                          ,seller_info.count,buyer_info.unfinished_task_penalty)
            X.append(X_i)
        X = np.array(X).T

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X,ys,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X,ys,buyer_info.V,buyer_info.a_val,seller_info.count,buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_penalty_history.append(buyerPenalties)

        # Run through sellers to update policy
        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []

        for j in range(0, seller_info.count):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].updateQ(actions, x_j, 0.,0.)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)

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
        'seller_policies': results_dir['seller_policies'],
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

    return eval_dict