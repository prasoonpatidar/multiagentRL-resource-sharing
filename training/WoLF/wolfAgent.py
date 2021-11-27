'''
This file contains Agent Class for WolfPHC learning algorithm
'''
import numpy as np
import os
import pickle

from training.seller_utils import action2y
from training.WoLF.wolf_utils import allSellerActions2stateIndex

class wolfAgent:

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
        self.seller_count = seller_count
        self.buyer_count = buyer_count
        self.y_min = aux_price_min
        self.y_max = aux_price_max
        self.policy_store = train_config.policy_store
        self.discount_factor = train_config.discount_factor
        self.learning_rate = train_config.learning_rate

        # get derived information
        self.__state_span = action_number**seller_count
        self.__action_size = action_number

        # create required variables for current state and current action
        # self.__current_state = np.random.randint(0, self.__state_span)
        self.__next_state = -1
        self.__action = 0
        # self.__y = -1
        self.__providedResources = [np.zeros(self.buyer_count)]
        self.__demandedResources = [np.zeros(self.buyer_count)]

        # create required containers for learner

        if not evaluate:
            # check if policy store exists
            if not os.path.exists(self.policy_store):
                os.makedirs(self.policy_store)

            self.__Q = np.zeros((self.__state_span,self.__action_size))
            self.__policy = np.ones((self.__state_span,self.__action_size)) \
                            * (1 / self.__action_size)
            self.__meanPolicy = np.ones((self.__state_span,self.__action_size)) \
                            * (1 / self.__action_size)
            self.__count = np.zeros(self.__state_span)
        else: #evaluate agent only
            # self.trainable=False
            self.__policy = pickle.load(open(f'{self.policy_store}/{self.id}_policy.pb', 'rb'))

    def get_next_action(self, curr_state):
        randomNumber = np.random.random()
        self.__action = 0
        while randomNumber >= self.__policy[curr_state][self.__action]:
            randomNumber -= self.__policy[curr_state][self.__action]
            self.__action += 1
        # self.__y = sellerAction2y(self.__action, self.__action_size,
        #                           self.y_min, self.y_max)
        return self.__action

    def get_policy(self):
        return self.__policy

    def Qmax(self):
        return max(self.__Q[self.__next_state])

    def updateQ(self, curr_state, curr_action, R, α, df):
        # update Q table
        self.__Q[curr_state][curr_action] = \
            (1 - α) * self.__Q[curr_state][curr_action] \
            + α * (R + df * self.Qmax())
        return None

    def updateMeanPolicy(self, curr_state):
        # update mean policy based on how many times a particular state is reached
        self.__count[curr_state] += 1
        self.__meanPolicy[curr_state] += (self.__policy[curr_state] - \
         self.__meanPolicy[curr_state]) / self.__count[curr_state]
        return None

    def updatePolicy(self, curr_state, lr_win):
        lr_lose = 50 * lr_win
        #check if seller lost or win this game
        if np.dot(self.__policy[curr_state], self.__Q[curr_state]) \
                > np.dot(self.__meanPolicy[curr_state], self.__Q[curr_state]):
            lr = lr_win
        else:
            lr = lr_lose
        # update policy based on based action from Q matrix
        bestAction = np.argmax(self.__Q[curr_state])
        for i in range(0, self.__action_size):
            if i == bestAction:
                continue
            delta = min(self.__policy[curr_state][i],
                    lr / (self.__action_size - 1))
            self.__policy[curr_state][i] -= delta
            self.__policy[curr_state][bestAction] += delta

    def getBuyerExperience(self, i):
        return np.mean([xr[i] for xr in self.__providedResources])

    def add_purchase_history(self, x_j, z_j):
        self.__demandedResources.append(x_j)
        self.__providedResources.append(z_j)