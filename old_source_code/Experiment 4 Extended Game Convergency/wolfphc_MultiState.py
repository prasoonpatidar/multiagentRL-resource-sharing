#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
备注：
在这个版本中，用于计算R的x_{i,j}可以取连续的数值，由下式计算得到。
in this version, we use the following formula to calculate x_{i,j}, which is a continuous variable
x_j = V * yAll[j] + A / d - np.e

状态：状态设置为上一时刻的卖家的联合动作
State: is defined as all the providers action at time t_1
"""

import numpy as np
import matplotlib.pyplot as plt 
import time 
import math
from scipy.optimize import minimize, LinearConstraint
import plotHistory as ph
    
def sellerAction2y(sellerAction,sellerActionSize,y_min,y_max):
    y = y_min + (y_max - y_min) / sellerActionSize * sellerAction
    return y

def allSellerActions2stateIndex(allSellerActions,N,sellerActionSize):
    stateIndex = 0
    for i in range(0,N):
        stateIndex = stateIndex * sellerActionSize + allSellerActions[i]
    return stateIndex

class Seller:
    def __init__(self,sellerIndex,selleractionSize,stateSize,c_j,y_min,y_max,M, max_resources,consumer_penalty_coeff, producer_penalty_coeff):
        #卖家的个人信息
        # basic information of the provider
        self.__c = c_j   #该卖家的成本系数 cost coefficient of the provider
        
        #卖家的编号、状态空间大小、动作空间大小
        # provider's index, size of state space, size of action space
        self.__sellerIndex = sellerIndex
        self.__stateSize = stateSize
        self.__actionSize = selleractionSize
        self.__y_min = y_min
        self.__y_max = y_max
        # new additions
        self.__consumer_penalty_coeff = consumer_penalty_coeff
        self.__producer_penalty_coeff = producer_penalty_coeff
        self.__buyerCount = M
        self.__sellerResourceCount = max_resources
        
        
        #当前状态、当前动作
        # current state, current action
        self.__currentState = np.random.randint(0,self.__stateSize)
        self.__nextState = -1
        self.__action = -1
        self.__y = -1

        # --Dispatching riders to drivers--

        #Q表、平均策略、策略、计数器
        # Q table, average policy, policy, index count, provided resources history
        self.__Q = np.zeros((self.__stateSize,self.__actionSize))
        self.__policy = np.ones((self.__stateSize,self.__actionSize)) \
                        * (1 / self.__actionSize)
        self.__meanPolicy = np.ones((self.__stateSize,self.__actionSize)) \
                        * (1 / self.__actionSize)
        self.__count = np.zeros(self.__stateSize)
        # self.__providedResources = {}
        self.__providedResources = [np.zeros(self.__buyerCount)]
        self.__demandedResources = [np.zeros(self.__buyerCount)]

    def max_resources(self):
        return self.__sellerResourceCount

    def show(self):
        print("\nself.__c =",self.__c)
        
        print("self.__sellerIndex =",self.__sellerIndex)
        print("self.__stateSize =",self.__stateSize)
        print("self.__actionSize =",self.__actionSize)
        print("self.__y_min =",self.__y_min)
        print("self.__y_max =",self.__y_max)
        
        print("self.__currentState =",self.__currentState)
        print("self.__nextState =",self.__nextState)
        print("self.__action =",self.__action)
        print("self.__y =",self.__y)
        
        print("self.__Q =\n",self.__Q)
        print("self.__policy =\n",self.__policy)
        print("self.__meanPolicy =\n",self.__meanPolicy)
        print("self.__count =\n",self.__count)
        print()
        
    def actionSelect(self):
        randomNumber = np.random.random()
#        print("self.__sellerIndex =",self.__sellerIndex)
#        print("randomNumber =",randomNumber)
        self.__action = 0
        while randomNumber >= self.__policy[self.__currentState][self.__action]:
            randomNumber -= self.__policy[self.__currentState][self.__action]
            self.__action += 1
        self.__y = sellerAction2y(self.__action,self.__actionSize,
                                  self.__y_min,self.__y_max)
#        print("self.__action =",self.__action,"\n")
        return self.__action
    
    def Qmax(self):
        return max(self.__Q[self.__nextState])

    def updateQ(self,allSellerActions,x_j,α,df,N,sellerActionSize): 
#        print("\nself.__sellerIndex =",self.__sellerIndex)
#        print("self.__currentState =",self.__currentState)
#        print("Q表更新前：")
# print before Q table update
#        print("self.__Q[self.__currentState] =",self.__Q[self.__currentState])
#        print("α =",α)
        
        #计算即时奖励R
        # calculate reward
        yAll = sellerAction2y(allSellerActions,sellerActionSize,
                              self.__y_min,self.__y_max)
        # R = self.__y / sum(yAll) * (1 / self.__y - self.__c) * sum(x_j)
        ##todo: Add a new reward function for sellers based on past demands as a Fx

        R = self.reward(yAll, x_j)

#        print("相应的R值 =",R) # R value
        
        #拿到下一时刻的状态 provider's state at t+1
        self.__nextState = allSellerActions2stateIndex(allSellerActions,\
                                                       N,sellerActionSize)
#        print("self.__nextState =",self.__nextState)
        
        #更新Q表 update Q table
#        print("self.Qmax() =",self.Qmax())
        self.__Q[self.__currentState][self.__action] = \
        (1 - α) * self.__Q[self.__currentState][self.__action] \
        + α * (R + df * self.Qmax())
#        print("Q表更新后：") # Q table updated
#        print("self.__Q[self.__currentState] =",self.__Q[self.__currentState])
        ## todo: Return new z values along with R
        return R, self.__providedResources[-1] #R是该卖家的效益函数的值 R is the revenue for p
        
    def updateMeanPolicy(self):
#        print("平均策略更新前：") # print- before update mean policy
#        print("self.__count[self.__currentState] =",self.__count[self.__currentState])
#        print("self.__meanPolicy[self.__currentState] =\n",self.__meanPolicy[self.__currentState])
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
        self.__count[self.__currentState] += 1
        self.__meanPolicy[self.__currentState] += \
        (self.__policy[self.__currentState] - \
         self.__meanPolicy[self.__currentState]) \
         / self.__count[self.__currentState]
#        print("平均策略更新后：") # print -mean policy updated
#        print("self.__count[self.__currentState] =",self.__count[self.__currentState])
#        print("self.__meanPolicy[self.__currentState] =\n",self.__meanPolicy[self.__currentState])
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
        
    def updatePolicy(self,δ_win):
        #print("\n策略更新前：self.__policy =",self.__policy)
        δ_lose = 50 * δ_win
#        print("\n卖家",self.__sellerIndex,":") # print provider index
#        print("策略更新前：") # print befor policy updated
#        print("δ_win =",δ_win,"δ_lose =",δ_lose)
#        print("self.__Q[self.__currentState] =\n",self.__Q[self.__currentState])
#        print("self.__action =",self.__action)
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
#        print("self.__meanPolicy[self.__currentState] =\n",self.__meanPolicy[self.__currentState])
#        r1 = np.dot(self.__policy[self.__currentState],self.__Q[self.__currentState])
#        r2 = np.dot(self.__meanPolicy[self.__currentState],self.__Q[self.__currentState])
#        print("r1 =",r1,"r2 =",r2)
        if np.dot(self.__policy[self.__currentState],self.__Q[self.__currentState]) \
        > np.dot(self.__meanPolicy[self.__currentState],self.__Q[self.__currentState]):
            δ = δ_win
#            print("δ = δ_win")
        else:
            δ = δ_lose
#            print("δ = δ_lose")
        
        bestAction = np.argmax(self.__Q[self.__currentState])
#        print("self.__sellerIndex =",self.__sellerIndex)
#        print("self.__currentState =",self.__currentState)
#        print("bestAction =",bestAction)
        for i in range(0,self.__actionSize):
            if i == bestAction:
                continue
            Δ = min(self.__policy[self.__currentState][i],
                    δ / (self.__actionSize - 1))
            self.__policy[self.__currentState][i] -= Δ
            self.__policy[self.__currentState][bestAction] += Δ
#        print("策略更新后：") # print policy updated
#        print("self.__Q[self.__currentState] =\n",self.__Q[self.__currentState])
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
        
        
    def updateState(self):
#        print("状态更新前：") # print before state update
#        print("self.__currentState =",self.__currentState)
        self.__currentState = self.__nextState
#        print("状态更新后：")
#        print("self.__currentState =",self.__currentState)
        
    def showPolicy(self):
        print("\n卖家",self.__sellerIndex,":") # print device index
        print("self.__meanPolicy =\n",self.__meanPolicy)
        print("self.__policy =\n",self.__policy)


    # Get experience of buyer(fz(i) from given seller
    def getBuyerExperience(self, i):
        return np.mean([xr[i] for xr in self.__providedResources])

    # reward function for given action, state and consumer requests

    def reward(self, yAll, x_j):

        deficit = np.maximum(0, np.sum(x_j) - self.max_resources())
        z_j=  x_j*(1-deficit/np.sum(x_j))

        # Update seller values
        self.__demandedResources.append(x_j)
        self.__providedResources.append(z_j)

        # Get reward value based on everything
        R = 0
        for i in range(0,self.__buyerCount):
            R += (self.__y/(np.sum(yAll))) * ( x_j[i]*(1/self.__y) - z_j[i]*self.__c)

        R += self.__producer_penalty_coeff*(np.sum(z_j) - self.max_resources())
        return R
        
class Record:
    def __init__(self,N,length):
        self.__index = 0
        self.__N = N
        self.__length = length
        self.__arr = np.array([[-1] * self.__N] * self.__length)#self.__arr是np.array类型的
    
    def isConverged(self,actions):
        self.__arr[self.__index] = actions#actions的类型是np.array actions are considered as np array
        self.__index = (self.__index + 1) % self.__length
        variance = np.var(self.__arr,axis = 0)
        if sum(variance) == 0:
            #画出arr plot the arr
            self.__arr = self.__arr.T
            iterations = range(0,self.__length)
            plt.figure()
            for j in range(0,self.__N):
                plt.plot(iterations,self.__arr[j],"o:",label = "seller %d"%(j + 1))
            plt.legend()
            plt.xlabel("the last %d iterations"%self.__length)
            plt.ylabel("actions")
            plt.savefig("收敛曲线.jpg",dpi = 300) # plot converge plot
            plt.show()
            
            return True
        else:
            return False

        
     
def wolfphc_MultiState(N,M,c,V,a,y_min,y_max,actionNumber,times, max_resources_per_seller,consumer_penalty_coeff, producer_penalty_coeff):
    #******（1）设置参数*************       parameter seting
    #Q表参数 Q table parameters
    df = 0.30 #discount factor,折扣因子。推荐：df ∈ [0.88,0.99]
    α = 1 / 3 #用于更新Q值的学习率 learning rate for updating the Q value


    #卖家参数 provider's parameters
    sellerActionSize = actionNumber  #卖家动作数 provider action
    stateSize = sellerActionSize ** N    
    
    #******（2）初始化卖家们*************  Initialize the providers
    #初始化卖家 provider initilization
    sellers = []
    # max_resource_per_seller = 10.0
    for j in range(0,N):
        tmpSeller = Seller(j,sellerActionSize,stateSize,c[j],y_min,y_max, M, max_resources_per_seller[j], consumer_penalty_coeff, producer_penalty_coeff)
        sellers.append(tmpSeller)
    
    #******（3）更新Q表、平均策略、策略************* Q table update, mean policy, policy
    pricesHistory = {}         #用于保存"所有的【卖家报价】"的历史记录  save the provider's histroy auxiliary price histroy
    purchasesHistory = {}       #用于保存"所有的【单个买家的总购买数量】"的历史记录 save each device's total resource demand history
    providedResourcesHistory = {}        # save each device's total resources accepted history
    sellerUtilitiesHistory = {}   #用于保存“所有的卖家的效益“的历史记录 save all the providers utility history
    buyerUtilitiesHistory = {}    #用于保存“所有的买家的效益“的历史记录 save all the devices utility history
    record = Record(N,500)#用于记录最近的连续500次的【所有卖家的动作的编号】.用于判断是否收敛 record the most recent 500 interations [all the devices' index], for convergency checking
    start = time.perf_counter()
    timeLimit_min = 1 #timeLimit_min是以分钟为单位的限定时间 the unit of timeLimit_min is minutes
    start_time = time.time()
    for t in range(0,times):
        if (t%100==0):
            print(f"Completed {t} iterations in {round(time.time()-start_time,3)} secs...")
            start_time = time.time()
        #参数 parameters
        δ_win = 1 / (500 + 0.1 * t)
        
        #获得联结动作 actions get all the providers action
        allSellerActions = []
        for tmpSeller in sellers:
            allSellerActions.append(tmpSeller.actionSelect())
        allSellerActions = np.array(allSellerActions)
        yAll = sellerAction2y(allSellerActions,sellerActionSize,y_min,y_max)
    
        #保存本次迭代的【所有卖家的报价】 save this interation's [all the devices' auxiliary price]
        prices = 1 / yAll
        pricesHistory[t] = prices
        
        #根据卖家的动作值y,换算出买家的购买数量
        ##todo: Buyer experience and purchase calculator
        # get the buyer experience with sellers based on previous purchases
        cumulativeBuyerExperience = np.zeros((M, N))
        for i in range(0,M):
            for j in range(0,N):
                cumulativeBuyerExperience[i][j] = sellers[j].getBuyerExperience(i)


        # get the amount of resources purchased by each device based on y
        X = []
        for i in range(0,M):
            X_i = buyerPurchaseCalculator(cumulativeBuyerExperience[i,:], yAll,V[i],a[i],N,consumer_penalty_coeff)
            X.append(X_i)
        X = np.array(X).T

        # X = []  #X是由数组x_j组成的数组
        # for j in range(0,N):
        #     x_j = V * yAll[j] + a - np.e #x_j是由【所有买家向第j个卖家购买的产品数量】组成的数组
        #     # x_j is a list describes the amount of resources purchased by each device from provider j
        #     X.append(x_j)
        # X = np.array(X)
        
        #保存本次迭代的【每个买家的总购买数量】
        #save [the total amount of resources purchased by each device], sum_X_ij of over N
        purchases = X.sum(axis = 0) #purchases是由【每个买家的总购买数量】组成的数组 purchases is the sum_X_ij of over N
        purchasesHistory[t] = purchases
        # providedResourcesHistory.append(cumulativeBuyerExperience.sum(axis=0))

        #保存本次迭代的【每个买家的效益】 save all the devices unitity in this interation
        #buyerUtilitiesCalculator(X,yAll,V,a,N,M)的返回值是由【每个买家的效益】组成的数组
        buyerUtilities = buyerUtilitiesCalculator(X,yAll,V,a,N,M, cumulativeBuyerExperience, consumer_penalty_coeff)
        buyerUtilitiesHistory[t] = buyerUtilities
        
        #更新Q表、平均策略、策略 update the Q table, mean policy, and policy
        sellerUtilities = []    #sellerUtilities是由【每个卖家的效益】组成的数组
        sellerProvidedResources = []
        # sellerUtilities is a list describes [each provider's utility]
        for j in range(0,N):
            # x_j = V * yAll[j] + a - np.e
            x_j = X[j]
            tmpSellerUtility, z_j = sellers[j].updateQ(allSellerActions,x_j,α,df,N,sellerActionSize)#更新Q表
            # todo: Update new z values along with R
            sellerUtilities.append(tmpSellerUtility)
            sellerProvidedResources.append(z_j)
            sellers[j].updateMeanPolicy()    #更新平均策略 update mean policy
            sellers[j].updatePolicy(δ_win)   #更新策略 update policy
            sellers[j].updateState()         #更新状态 update state
        sellerUtilities = np.array(sellerUtilities)
        sellerUtilitiesHistory[t]=sellerUtilities

        ##todo: Update new z values(resourceProvidedHistory) along with sellerUtilityHistory
        sellerProvidedResources = np.array(sellerProvidedResources)
        providedResourcesHistory[t] = sellerProvidedResources

#        #判断是否已经收敛
#        #判断标准，如果在【最近的连续500次迭代】中，【所有卖家的报价】保持不变，则认为是已经收敛
#        if record.isConverged(allSellerActions) == True:
#            break
#        #判断是否超出限定时间。如果超出，则返回False。
#        stop = time.perf_counter()
#        if (stop - start) / 60.0 > timeLimit_min:
#            return False
    #Wolfphc算法结束
    # todo: Add resourceProvidedHistory along with sellerUtilityHistory
    return pricesHistory,purchasesHistory, providedResourcesHistory,times
    
#    #需要展示的数据:
#    #画出每个卖家的price的变化
#    plt.subplot(2,2,2)
#    ph.plotHistory(pricesHistory,times,None,None,"provider")
#    #画出每个买家的总购买数量的变化
#    plt.subplot(2,2,4)
#    ph.plotHistory(purchasesHistory,times,"Iterations during RLPM",None,"device")
#    #画出每个卖家的效益的变化
#    ph.plotHistory(sellerUtilitiesHistory,times,N,M,"Provider’s utility (WoLF-PHC)","provider","WoLF-PHC-sellerUtilities")
#    #画出每个买家的效益的变化
#    ph.plotHistory(buyerUtilitiesHistory,times,N,M,"IoT device’s utility (WoLF-PHC)","IoT device","WoLF-PHC-buyerUtilities")
    
    
#    #返回True
#    return True
    

def buyerUtilitiesCalculator(X,yAll,V,a,N,M, cumulativeBuyerExperience, consumer_penalty_coeff):#在已知购买数量和定价的情况下，计算出所有买家的效益
    #输入参数介绍： the input
    #X是由数组x_j组成的数组。x_j是由【所有买家向第j个卖家购买的产品数量】组成的数组
    # X is a list for all the x_j. x_j is a list that describes [the number of resources all the devices purchased from provider j]
    #yAll是由动作值y组成的数组
    #yAll is a list for al the possible ys
    #V是np.array
    #V is np.array
    #a是np.array
    # a is np.array
    buyerUtilities = []
    for i in range(0,M):#计算第i个买家的效益 get device i's utility
        buyerUtility = 0
        for j in range(0,N):
            buyerUtility += (V[i] * math.log(X[j][i] - a[i] + np.e) \
                             - X[j][i] / yAll[j]) * (yAll[j] / sum(yAll))\
                            - consumer_penalty_coeff * (cumulativeBuyerExperience[i][j] - X[j][i])**2
            # todo: Add the regularizer based on Z values
        buyerUtilities.append(buyerUtility)
    buyerUtilities = np.array(buyerUtilities)
    return buyerUtilities


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
