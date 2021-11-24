'''
This is the main RL Algorithm file for WolfPHC. it contains wrapper functions to learn and evaluate RL policies.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import logging

# custom project libraries
from training.wolfPHC.run_helper import buyerPenaltiesCalculator, buyerUtilitiesCalculator
from training.wolfPHC.run_helper import logger_handle, initialize_agent, get_ys, choose_prob, cumlativeBuyerExp, getPurchases

'''
Main learner function to run policy for sellers using WoLF-PHC agent
'''

def learn_policy(run_config, seller_info, buyer_info, train_config, logger_pass):
    # Initialize the logger
    logger = logger_handle(logger_pass)

    # get required parameters for WolFPHC algorithm
    discount_factor = train_config.discount_factor
    learning_rate = train_config.learning_rate
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers, logger = initialize_agent(seller_info, buyer_info, train_config, logger)

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
        lr_win = 1 / (500 + 0.1 * train_iter)

        # get the prices for all seller agents
        ys, all_seller_actions = get_ys(sellers, train_config, seller_info)

        # print(ys, '==', train_iter)
        probAll, yAll = choose_prob(ys, compare=False, yAll=None)
        # Save prices in history
        prices = 1 / ys
        price_history.append(prices)

        cumulativeBuyerExperience = cumlativeBuyerExp(buyer_info, sellers)
        X = getPurchases(buyer_info, cumulativeBuyerExperience, ys, probAll)

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, probAll,
                                                  buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, buyer_info.count,
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
            sellers[j].updateState()
