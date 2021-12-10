'''
This file contains Agent Class for Q learning algorithm
'''

import numpy as np
import os
import pickle


# custom libraries

class qAgent:
    # constructor for wolfphcAgent
    def __init__(self, seller_id, max_resources, cost_per_unit, action_number,
                 aux_price_min, aux_price_max, seller_idle_penalty, seller_count,
                 buyer_count, train_config, evaluate=False):

        # get basic information
        self.cost_per_unit = cost_per_unit
        self.id = seller_id
        self.max_resources = max_resources
        self.action_number = action_number
        self.penalty_coeff = seller_idle_penalty
        self.discount_factor = train_config.discount_factor
        self.learning_rate = train_config.learning_rate
        self.policy_store = train_config.policy_store
        self.explore_prob = train_config.explore_prob
        self.buyer_count = buyer_count
        self.y_min = aux_price_min
        self.y_max = aux_price_max

        # get derived information
        self.__state_size = action_number ** seller_count
        self.__action_size = action_number

        self.__Q = np.zeros(self.action_number)
        self.__policy = np.array([1 / self.action_number] \
                                 * self.action_number)
        # create required variables for current state and current action
        self.__action = 0
        # self.__y = action2y(self.__action,self.action_number,self.y_min,self.y_max)
        self.__providedResources = [np.full(self.buyer_count, 0.)]
        self.__demandedResources = [np.full(self.buyer_count, 0.)]

        if not evaluate:
            if not os.path.exists(self.policy_store):
                os.makedirs(self.policy_store)
        else:
            # Initialize policy for evaluation
            seller_policy = pickle.load(open(f'{self.policy_store}/{self.id}_policy.pb', 'rb'))
            self.__policy = seller_policy

    def get_next_action(self):
        randomNumber = np.random.random()
        self.__action = 0
        while randomNumber >= self.__policy[self.__action]:
            randomNumber -= self.__policy[self.__action]
            self.__action += 1
        return self.__action

    def Qmax(self):
        return max(self.__Q)

    def updateQ(self, R, α, df):
        # Update Q
        self.__Q[self.__action] = (1 - α) * self.__Q[self.__action] \
                                  + α * (R + df * self.Qmax())

    def get_policy(self):
        return self.__policy

    def add_purchase_history(self, x_j, z_j):
        self.__demandedResources.append(x_j)
        self.__providedResources.append(z_j)

    def updatePolicy(self, ε):
        for i in range(0, self.action_number):
            self.__policy[i] = ε / self.action_number
        bestAction = np.argmax(self.__Q)
        self.__policy[bestAction] += 1 - ε

    def getBuyerExperience(self, i):
        return np.mean([xr[i] for xr in self.__providedResources[-100:]])
