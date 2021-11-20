'''
This file contains Agent Class for Q learning algorithm
'''

import numpy as np

#custom libraries
from training.QLearning.utils import action2y

class qAgent:
    # constructor for wolfphcAgent
    def __init__(self, seller_id, max_resources, cost_per_unit, action_number,
                 aux_price_min, aux_price_max, seller_idle_penalty, seller_count,
                 buyer_count, seller_policy=None, is_trainer=True):
        # get basic information
        self.cost_per_unit = cost_per_unit
        self.id = seller_id
        self.max_resources = max_resources
        self.action_number = action_number
        self.penalty_coeff = seller_idle_penalty
        self.seller_count = seller_count
        self.buyer_count = buyer_count
        self.y_min = aux_price_min
        self.y_max = aux_price_max

        # get derived information
        self.__state_size = action_number ** seller_count
        self.__action_size = action_number

        # create required containers for learner
        if is_trainer:
            self.trainable = True
            self.__Q = np.zeros(self.action_number)
            self.__policy = np.array([1 / self.action_number] \
                                 * self.action_number)
        else:  # evaluate agent only
            self.trainable = False
            self.__policy = seller_policy

        # create required variables for current state and current action
        self.__action = 0
        self.__y = action2y(self.__action,self.action_number,self.y_min,self.y_max)

        self.__providedResources = [np.zeros(self.buyer_count)]
        self.__demandedResources = [np.zeros(self.buyer_count)]

    def get_next_action(self):
        randomNumber = np.random.random()
        self.__action = 0
        while randomNumber >= self.__policy[self.__action]:
            randomNumber -= self.__policy[self.__action]
            self.__action += 1
        self.__y = action2y(self.__action,self.action_number,self.y_min,self.y_max)
        return self.__action

    def Qmax(self):
        return max(self.__Q)

    def updateQ(self,actions,x_j,α,df):
        ys = action2y(actions,self.action_number,self.y_min,self.y_max)
        R, utility, penalty = self.reward(actions, x_j)
        # Update Q

        if self.trainable:
            self.__Q[self.__action] = (1 - α) * self.__Q[self.__action] \
                                      + α * (R + df * self.Qmax())
        return utility, penalty, self.__providedResources[-1]

    def reward(self, ys, x_j):

        deficit = np.maximum(0, np.sum(x_j) - self.max_resources)
        z_j=  x_j*(1-deficit/np.sum(x_j))

        # Update seller values
        self.__demandedResources.append(x_j)
        self.__providedResources.append(z_j)

        # Get reward value based on everything
        R = 0
        for i in range(0,self.buyer_count):
            R += (self.__y/(np.sum(ys))) * ( x_j[i]*(1/self.__y) - z_j[i] * self.cost_per_unit)

        utility = R
        penalty = self.penalty_coeff*(np.sum(z_j) - self.max_resources)
        R += penalty
        return R, utility, penalty

    def get_policy(self):
        return self.__policy

    def updatePolicy(self,ε):
        for i in range(0,self.action_number):
            self.__policy[i] = ε / self.action_number
        bestAction = np.argmax(self.__Q)
        self.__policy[bestAction] += 1 - ε

    def getBuyerExperience(self, i):
        return np.mean([xr[i] for xr in self.__providedResources])
