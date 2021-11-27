'''
Functions to evaluate buyer's fixed strategies
'''

import math
import numpy as np
from scipy.optimize import minimize

def getBuyerExperience(sellers, buyer_info):
    # get the buyer experience with sellers based on previous purchases
    num_sellers = len(sellers)
    num_buyers = buyer_info.count
    cumulativeBuyerExperience = np.zeros((num_buyers, num_sellers))
    for i in range(0, num_buyers):
        for j in range(0, num_sellers):
            cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)
    return cumulativeBuyerExperience

def get_buyer_rewards(X, ys, probAll, cumulativeBuyerExperience, buyer_info):
    # get buyer utilities
    buyer_utilities = buyerUtilitiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, probAll,
                                                  buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)

    # get buyer penalties
    buyer_penalties = buyerPenaltiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)

    return buyer_utilities, buyer_penalties


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
def buyerPurchaseCalculator(cumulativeBuyerExperience, yAll, V_i, a_i, y_prob, consumer_penalty_coeff):
    # get singleBuyer utility function to maximize
    N = len(y_prob)

    def singleBuyerUtilityFunction(x_i):
        buyerUtility = 0.
        for j in range(0, N):
            buyerUtility += (V_i * math.log(x_i[j] - a_i + np.e) \
                             - x_i[j] / yAll[j]) * y_prob[j] \
                            - consumer_penalty_coeff * (cumulativeBuyerExperience[j] - x_i[j]) ** 2
        return -1 * buyerUtility

    # solve optimization function for each buyer
    xi_opt_sol = minimize(singleBuyerUtilityFunction, np.zeros(N), bounds=[(0, 100)] * N)

    x_opt = xi_opt_sol.x
    return x_opt


# Buyer Utilities Calculator
def buyerUtilitiesCalculator(X, yAll, V, a, y_prob, M, cumulativeBuyerExperience, consumer_penalty_coeff):
    N = len(y_prob)
    buyerUtilities = []
    for i in range(0, M):
        buyerUtility = 0
        for j in range(0, N):
            buyerUtility += (V[i] * math.log(X[j][i] - a[i] + np.e) \
                             - X[j][i] / yAll[j]) * y_prob[j]
            # todo: Add the regularizer based on Z values
        buyerUtilities.append(buyerUtility)
    buyerUtilities = np.array(buyerUtilities)
    return buyerUtilities


# Buyer Penalties Calculator
def buyerPenaltiesCalculator(X, yAll, V, a, M, cumulativeBuyerExperience, consumer_penalty_coeff):
    N = len(yAll)
    buyerPenalties = []
    for i in range(0, M):
        buyerPenalty = 0
        for j in range(0, N):
            buyerPenalty += consumer_penalty_coeff * (cumulativeBuyerExperience[i][j] - X[j][i]) ** 2
            # todo: Add the regularizer based on Z values
        buyerPenalties.append(buyerPenalty)
    buyerPenalties = np.array(buyerPenalties)
    return buyerPenalties