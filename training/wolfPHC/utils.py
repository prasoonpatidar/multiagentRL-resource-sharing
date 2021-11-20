'''
Utility fuctions for WolfPHC
'''

import math
import numpy as np
from scipy.optimize import minimize

# convert seller action to auxilarry price
def sellerAction2y(sellerAction, sellerActionSize, y_min, y_max):
    y = y_min + (y_max - y_min) / sellerActionSize * sellerAction
    return y

# Get next state is based on actions from last round
def allSellerActions2stateIndex(allSellerActions,N,sellerActionSize):
    stateIndex = 0
    for i in range(0,N):
        stateIndex = stateIndex * sellerActionSize + allSellerActions[i]
    return stateIndex

# Buyer Purchase Calculator
def buyerPurchaseCalculator(cumulativeBuyerExperience, yAll,V_i,a_i,N, consumer_penalty_coeff):
    # get singleBuyer utility function to maximize
    def singleBuyerUtilityFunction(x_i):
        buyerUtility = 0.
        for j in range(0, N):
            buyerUtility += (V_i * math.log(x_i[j] - a_i + np.e) \
                             - x_i[j] / yAll[j]) * (yAll[j] / sum(yAll)) \
                            - consumer_penalty_coeff * (cumulativeBuyerExperience[j] - x_i[j])**2
        return -1*buyerUtility
    # solve optimization function for each buyer
    xi_opt_sol = minimize(singleBuyerUtilityFunction, np.zeros(N), bounds=[(0,100)]*N)

    x_opt = xi_opt_sol.x
    return x_opt

# Buyer Utilities Calculator
def buyerUtilitiesCalculator(X,yAll,V,a,N,M, cumulativeBuyerExperience, consumer_penalty_coeff):
    buyerUtilities = []
    for i in range(0,M):
        buyerUtility = 0
        for j in range(0,N):
            buyerUtility += (V[i] * math.log(X[j][i] - a[i] + np.e) \
                             - X[j][i] / yAll[j]) * (yAll[j] / sum(yAll))
            # todo: Add the regularizer based on Z values
        buyerUtilities.append(buyerUtility)
    buyerUtilities = np.array(buyerUtilities)
    return buyerUtilities


# Buyer Penalties Calculator
def buyerPenaltiesCalculator(X,yAll,V,a,N,M, cumulativeBuyerExperience, consumer_penalty_coeff):
    buyerPenalties = []
    for i in range(0,M):
        buyerPenalty = 0
        for j in range(0,N):
            buyerPenalty += consumer_penalty_coeff * (cumulativeBuyerExperience[i][j] - X[j][i])**2
            # todo: Add the regularizer based on Z values
        buyerPenalties.append(buyerPenalty)
    buyerPenalties = np.array(buyerPenalties)
    return buyerPenalties
