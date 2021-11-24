'''
This contains helper functions to run the Q learning algorithm.
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

def logger_handle(logger_pass):
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('Q_learn_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    return logger

def initialize_agent( seller_info, buyer_info, train_config,
                      logger, compare=False, agentNum=None,
                       is_trainer=True, results_dir=None):

    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers = []
    action_number = train_config.action_count
    if compare:
        # when compare, we can generate any number of agents
        for seller_id in range(agentNum):
            max_resources = seller_info.max_resources[seller_id]
            cost_per_unit = seller_info.per_unit_cost[seller_id]
            tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
                                     aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                                     buyer_info.count)
            sellers.append(tmpSeller)
            logger.info(f"Initialized {agentNum} seller agents for compare")
    if is_trainer:
        for seller_id in range(seller_info.count):
            max_resources = seller_info.max_resources[seller_id]
            cost_per_unit = seller_info.per_unit_cost[seller_id]
            tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
                                 aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                                 buyer_info.count)
            sellers.append(tmpSeller)
            logger.info(f"Initialized {seller_info.count} seller agents for training")
    else:
        for seller_id in range(seller_info.count):
            seller_policy = results_dir['seller_policies'][seller_id]
            max_resources = seller_info.max_resources[seller_id]
            cost_per_unit = seller_info.per_unit_cost[seller_id]
            tmpSeller = qAgent(seller_id, max_resources, cost_per_unit, action_number,
                               aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                               buyer_info.count, seller_policy, is_trainer=False)
            sellers.append(tmpSeller)
        logger.info(f"Initialized {seller_info.count} seller agents for evaluation")
    return sellers, logger

def get_ys(sellers, train_config, seller_info):
    # get required parameters for Q-learning algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_config.action_count
    # get actions from all sellers
    actions = []
    for tmpSeller in sellers:
        actions.append(tmpSeller.get_next_action())
    actions = np.array(actions)
    ys = action2y(actions, action_number, aux_price_min, aux_price_max)
    return ys

def choose_prob(ys, compare=False, yAll=None):
    probAll = []
    if compare:
        for j in range(0, len(ys)):
            prob = ys[j] / sum(yAll)
            probAll.append(prob)
        yAll = yAll
    else:
        for j in range(0, len(ys)):
            prob = ys[j]/sum(ys)
            probAll.append(prob)
        yAll = ys
    return probAll, yAll

def cumlativeBuyerExp(buyer_info, sellers):
    # get the buyer experience with sellers based on previous purchases
    num_sellers = len(sellers)
    num_buyers = buyer_info.count
    cumulativeBuyerExperience = np.zeros((num_buyers, num_sellers))
    for i in range(0, num_buyers):
        for j in range(0, num_sellers):
            cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)
    return cumulativeBuyerExperience

def getPurchases(buyer_info, cumulativeBuyerExperience, ys, probAll):
    # get the amount of resources purchased by each device based on y
    X = []
    for i in range(0, buyer_info.count):
        X_i = buyerPurchaseCalculator(cumulativeBuyerExperience[i, :], ys, buyer_info.V[i], buyer_info.a_val[i]
                                      , probAll, buyer_info.unfinished_task_penalty)
        X.append(X_i)
    X = np.array(X).T
    return X

