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
from training.wolfPHC.utils import sellerAction2y


def logger_handle(logger_pass, train=True):
    logger_pass = dict(logger_pass)
    if train:
        logger_base = logger_pass.get('logger_base').getChild('WoLF_learn_policy')
        logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
        logger_pass['logger_base'] = logger_base
    else:
        logger_base = logger_pass.get('logger_base').getChild('WoLF_evaluate_policy')
        logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
        logger_pass['logger_base'] = logger_base
    return logger

def get_params(seller_info, train_config, logger):
    # get required parameters for WolFPHC algorithm
    discount_factor = train_config.discount_factor
    learning_rate = train_config.learning_rate
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    action_number = train_config.action_count
    logger.info("Fetched raw market information..")
    return discount_factor, learning_rate, aux_price_max, aux_price_min, action_number

def initialize_agent( seller_info, buyer_info, train_config,
                      logger, compare=False, agentNum=None,
                       is_trainer=True, results_dir=None, seller_id=None):
    discount_factor, learning_rate, aux_price_max, aux_price_min,\
    action_number = get_params(seller_info, train_config, logger)
    # initialize seller agents
    sellers = []

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
    elif is_trainer:
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
    ys = sellerAction2y(actions, action_number, aux_price_min, aux_price_max)
    return ys, actions, action_number

def choose_prob(ys, compare=False, yAll=None):
    probAll = []
    if compare:
        for j in range(0, len(ys)):
            prob = ys[j] / sum(yAll)
            probAll.append(prob)
    else:
        for j in range(0, len(ys)):
            prob = ys[j]/sum(ys)
            probAll.append(prob)
    return probAll

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

def get_lr(train_iter):
    # loop parameters
    lr_win = 1 / (500 + 0.1 * train_iter)
    return lr_win

def evaluation(sellers, train_config, all_seller_actions, yAll, X,train_iter, seller_info, logger, train=True):
    discount_factor, learning_rate, aux_price_max, aux_price_min, action_number = get_params(
        seller_info, train_config, logger)
    lr_win = get_lr(train_iter)
    # Run through sellers to update policy
    seller_utilities = []
    seller_penalties = []
    seller_provided_resources = []
    num_seller = len(sellers)
    if train:
        for j in range(0, num_seller):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].updateQ(all_seller_actions, x_j, learning_rate,
                                                                         discount_factor, num_seller,
                                                                         action_number, yAll)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)
            sellers[j].updateMeanPolicy()  # 更新平均策略 update mean policy
            sellers[j].updatePolicy(lr_win)  # 更新策略 update policy
            sellers[j].updateState()  # 更新状态 update state
    else:
        for j in range(0, num_seller):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].updateQ(all_seller_actions, x_j, learning_rate,
                                                                         discount_factor, num_seller,
                                                                         action_number, yAll)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)
            sellers[j].updateState()  # update state
    return seller_utilities, seller_penalties, seller_provided_resources

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

