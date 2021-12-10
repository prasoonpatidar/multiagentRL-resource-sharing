'''
Helper functions for sellers in the market
'''

import numpy as np

def action2y(action,actionNumber,y_min,y_max):
    y = y_min + (y_max - y_min) / actionNumber * action
    return y

def ydiff2action(ys,actionNumber,y_min,y_max):
    action = np.floor((ys * actionNumber)/(y_max - y_min))
    return action

def choose_prob(ys, compare=False, yAll=None, prob_func='total'):
    probAll = []
    if compare:
        for j in range(0, len(ys)):
            if prob_func=='softmax':
                prob = np.exp(ys[j]) / sum(np.exp(yAll))
            else:
                prob = ys[j] / sum(yAll)
            probAll.append(prob)
        yAll = yAll
    else:
        for j in range(0, len(ys)):
            if prob_func == 'softmax':
                prob = np.exp(ys[j]) / sum(np.exp(ys))
            else:
                prob = ys[j] / sum(ys)
            probAll.append(prob)
        yAll = ys
    return np.array(probAll), np.array(yAll)


def get_rewards(sellers, X, yAll, probAll):

    seller_utilties,seller_penalties, seller_distributed_resources  = [],[],[]
    for j in range(len(sellers)):
        # get x_j vals for given seller
        x_j = X[j]
        deficit = np.maximum(0, np.sum(x_j) - sellers[j].max_resources)
        if np.sum(x_j)==0:
            z_j = np.zeros_like(x_j)
        else:
            z_j = x_j * (1 - deficit / np.sum(x_j))

        # Get reward value based on everything
        utility = 0
        y = yAll[sellers[j].id]
        for i in range(0, sellers[j].buyer_count):
            utility += (probAll[j]) * (z_j[i] * (1 / y) - z_j[i] * sellers[j].cost_per_unit)
        penalty = sellers[j].penalty_coeff * (np.sum(z_j) - sellers[j].max_resources)
        seller_utilties.append(utility)
        seller_penalties.append(penalty)
        seller_distributed_resources.append(z_j)

    return np.array(seller_utilties),np.array(seller_penalties), np.array(seller_distributed_resources)
