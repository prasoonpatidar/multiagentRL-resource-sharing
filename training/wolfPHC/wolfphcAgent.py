'''
This file contains Agent Class for WolfPHC learning algorithm
'''

import numpy as np


from training.wolfPHC.utils import action2y, allSellerActions2stateIndex


class wolfphcAgent:

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
        self.__state_size = action_number**seller_count
        self.__action_size = action_number

        # create required variables for current state and current action
        self.__current_state = np.random.randint(0, self.__state_size)
        self.__next_state = -1
        self.__action = -1
        self.__y = -1

        # create required containers for learner
        if is_trainer:
            self.trainable = True
            self.__Q = np.zeros((self.__state_size,self.__action_size))
            self.__policy = np.ones((self.__state_size,self.__action_size)) \
                            * (1 / self.__action_size)
            self.__meanPolicy = np.ones((self.__state_size,self.__action_size)) \
                            * (1 / self.__action_size)
            self.__count = np.zeros(self.__state_size)
        else: #evaluate agent only
            self.trainable=False
            self.__policy = seller_policy
        self.__providedResources = [np.zeros(self.buyer_count)]
        self.__demandedResources = [np.zeros(self.buyer_count)]

    def get_next_action(self):
        randomNumber = np.random.random()
        self.__action = 0
        while randomNumber >= self.__policy[self.__current_state][self.__action]:
            randomNumber -= self.__policy[self.__current_state][self.__action]
            self.__action += 1
        self.__y = action2y(self.__action, self.__action_size,
                                  self.y_min, self.y_max)
        return self.__action

    def get_policy(self):
        return self.__policy

    def Qmax(self):
        return max(self.__Q[self.__next_state])

    def updateQ(self, allSellerActions, x_j, α, df, N, sellerActionSize):
        # calculate reward
        yAll = action2y(allSellerActions, sellerActionSize,
                              self.y_min, self.y_max)

        R, utility, penalty = self.reward(yAll, x_j)

        # provider's state at t+1
        self.__next_state = allSellerActions2stateIndex(allSellerActions, \
                                                       N, sellerActionSize)
        #        print("self.__next_state =",self.__next_state)

        # update Q table
        if self.trainable:
            self.__Q[self.__current_state][self.__action] = \
                (1 - α) * self.__Q[self.__current_state][self.__action] \
                + α * (R + df * self.Qmax())
        return utility, penalty, self.__providedResources[-1]  # R是该卖家的效益函数的值 R is the revenue for p


    def reward(self, yAll, x_j):

        deficit = np.maximum(0, np.sum(x_j) - self.max_resources)
        z_j=  x_j*(1-deficit/np.sum(x_j))

        # Update seller values
        self.__demandedResources.append(x_j)
        self.__providedResources.append(z_j)

        # Get reward value based on everything
        R = 0
        for i in range(0,self.buyer_count):
            R += (self.__y/(np.sum(yAll))) * ( x_j[i]*(1/self.__y) - z_j[i] * self.cost_per_unit)

        utility = R
        penalty = self.penalty_coeff*(np.sum(z_j) - self.max_resources)
        R += penalty
        return R, utility, penalty


    def updateMeanPolicy(self):
        # update mean policy based on how many times a particular state is reached
        self.__count[self.__current_state] += 1
        self.__meanPolicy[self.__current_state] += (self.__policy[self.__current_state] - \
         self.__meanPolicy[self.__current_state]) / self.__count[self.__current_state]
        return None

    def updatePolicy(self, lr_win):
        lr_lose = 50 * lr_win
        #check if seller lost or win this game
        if np.dot(self.__policy[self.__current_state], self.__Q[self.__current_state]) \
                > np.dot(self.__meanPolicy[self.__current_state], self.__Q[self.__current_state]):
            lr = lr_win
        else:
            lr = lr_lose
        # update policy based on based action from Q matrix
        bestAction = np.argmax(self.__Q[self.__current_state])
        for i in range(0, self.__action_size):
            if i == bestAction:
                continue
            delta = min(self.__policy[self.__current_state][i],
                    lr / (self.__action_size - 1))
            self.__policy[self.__current_state][i] -= delta
            self.__policy[self.__current_state][bestAction] += delta

    def updateState(self):
        if self.trainable: # update happens at Q state
            self.__current_state = self.__next_state

    def getBuyerExperience(self, i):
        return np.mean([xr[i] for xr in self.__providedResources])
