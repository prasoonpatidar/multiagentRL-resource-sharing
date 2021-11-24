'''
This contains helper functions to run the Wolf-PHC learning algorithm.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import logging
from scipy.optimize import minimize, LinearConstraint

# custom libraries
from training.wolfPHC.wolfphcAgent import wolfphcAgent
from training.wolfPHC.utils import action2y


def logger_handle(logger_pass):
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('WoLF_learn_policy')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base
    return logger

def initialize_agent( seller_info, buyer_info, train_config,
                      logger, compare=False, agentNum=None,
                       is_trainer=True, results_dir=None, seller_id=None):

    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers = []
    action_number = train_config.action_count
    if compare:
        # when compare, we can generate any number of agents
        seller_policy = results_dir['seller_policies'][seller_id]
        max_resources = seller_info.max_resources[seller_id]
        cost_per_unit = seller_info.per_unit_cost[seller_id]
        tmpSeller = wolfphcAgent(seller_id, max_resources, cost_per_unit, action_number,
                                 aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                                 buyer_info.count, seller_policy, is_trainer=False)
        sellers.append(tmpSeller)
        logger.info(f"Initialized {agentNum} seller agents for compare")
    if is_trainer:
        for seller_id in range(seller_info.count):
            max_resources = seller_info.max_resources[seller_id]
            cost_per_unit = seller_info.per_unit_cost[seller_id]
            tmpSeller = wolfphcAgent(seller_id, max_resources, cost_per_unit, action_number,
                                 aux_price_min, aux_price_max, seller_info.idle_penalty, seller_info.count,
                                 buyer_info.count)
            sellers.append(tmpSeller)
            logger.info(f"Initialized {seller_info.count} seller agents for training")
    else:
        for seller_id in range(seller_info.count):
            seller_policy = results_dir['seller_policies'][seller_id]
            max_resources = seller_info.max_resources[seller_id]
            cost_per_unit = seller_info.per_unit_cost[seller_id]
            tmpSeller = wolfphcAgent(seller_id, max_resources, cost_per_unit, action_number,
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
    return ys, actions, action_number

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

# Buyer Purchase Calculator
def buyerPurchaseCalculator(cumulativeBuyerExperience, yAll,V_i,a_i,y_prob, consumer_penalty_coeff):
    # get singleBuyer utility function to maximize
    N = len(y_prob)
    def singleBuyerUtilityFunction(x_i):
        buyerUtility = 0.
        for j in range(0, N):
            buyerUtility += (V_i * math.log(x_i[j] - a_i + np.e) \
                             - x_i[j] / yAll[j]) * y_prob[j] \
                            - consumer_penalty_coeff * (cumulativeBuyerExperience[j] - x_i[j])**2
        return -1*buyerUtility
    # solve optimization function for each buyer
    xi_opt_sol = minimize(singleBuyerUtilityFunction, np.zeros(N), bounds=[(0,100)]*N)

    x_opt = xi_opt_sol.x
    return x_opt

# Buyer Utilities Calculator
def buyerUtilitiesCalculator(X,yAll,V,a,y_prob,M, cumulativeBuyerExperience, consumer_penalty_coeff):
    N = len(y_prob)
    buyerUtilities = []
    for i in range(0,M):
        buyerUtility = 0
        for j in range(0,N):
            buyerUtility += (V[i] * math.log(X[j][i] - a[i] + np.e) \
                             - X[j][i] / yAll[j]) * y_prob[j]
            # todo: Add the regularizer based on Z values
        buyerUtilities.append(buyerUtility)
    buyerUtilities = np.array(buyerUtilities)
    return buyerUtilities


# Buyer Penalties Calculator
def buyerPenaltiesCalculator(X,yAll,V,a,M, cumulativeBuyerExperience, consumer_penalty_coeff):
    N = len(yAll)
    buyerPenalties = []
    for i in range(0,M):
        buyerPenalty = 0
        for j in range(0,N):
            buyerPenalty += consumer_penalty_coeff * (cumulativeBuyerExperience[i][j] - X[j][i])**2
            # todo: Add the regularizer based on Z values
        buyerPenalties.append(buyerPenalty)
    buyerPenalties = np.array(buyerPenalties)
    return buyerPenalties

