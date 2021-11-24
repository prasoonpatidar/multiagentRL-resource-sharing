#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:14:51 2020

@author: xuhuiying
"""
import numpy as np
import matplotlib.pyplot as plt
import plotHistory as ph

# transform the action index to action y
def action2y(action,actionNumber,y_min,y_max):#把动作的编号转换成对应的动作值y
    y = y_min + (y_max - y_min) / actionNumber * action
    return y  
    
class Seller:
    def __init__(self,sellerIndex,actionNumber,cost,y_min,y_max):
        #卖家的基本信息
        # basic information of the provider
        self.__sellerIndex = sellerIndex
        self.__c = cost                                #该卖家的成本系数 cost coefficient of the provider
        self.__actionNumber = actionNumber
        self.__y_min = y_min
        self.__y_max = y_max
        
        #用于wolfphc算法的数据结构
        # data structure for wolfphc algorithm
        self.__Q = np.zeros(self.__actionNumber)
        #----confuse about the policy size and value
        self.__policy = np.array([1 / self.__actionNumber] \
                                 * self.__actionNumber)
        
        #该卖家当前的动作
        # provider's action at time t (current action)
        self.__action = 0
        self.__y = action2y(self.__action,self.__actionNumber,self.__y_min,self.__y_max)
        
        #该卖家以往所有的动作
        # provider's action at t+1
        self.__policyHistory = [self.__policy.tolist()]
        
    def show(self):
        print("\nself.__sellerIndex =",self.__sellerIndex)
        print("self.__c =",self.__c)
        print("self.__actionNumber =",self.__actionNumber)
        print("self.__y_min =",self.__y_min)
        print("self.__y_max =",self.__y_max)
        print("self.__Q =",self.__Q)
        print("self.__policy =",self.__policy)
        print("self.__action =",self.__action)
        print("self.__y =",self.__y)
        print("self.__policyHistory = ",self.__policyHistory,end = '\n\n')

    def actionSelect(self): # --confuse
        randomNumber = np.random.random() # [0/0. 1.0)
        self.__action = 0
        while randomNumber >= self.__policy[self.__action]:
            randomNumber -= self.__policy[self.__action]
            self.__action += 1
        self.__y = action2y(self.__action,self.__actionNumber,self.__y_min,self.__y_max)
        return self.__action
    
    def Qmax(self):
        return max(self.__Q)

    def updateQ(self,actions,x_j,α,df):
        ys = action2y(actions,self.__actionNumber,self.__y_min,self.__y_max)
        R = (self.__y / sum(ys)) * (1 / self.__y - self.__c) * sum(x_j)
        self.__Q[self.__action] = (1 - α) * self.__Q[self.__action] \
        + α * (R + df * self.Qmax())
        
    def updatePolicy(self,ε): #-- confuse += 1 - ε
        for i in range(0,self.__actionNumber):
            self.__policy[i] = ε / self.__actionNumber
        bestAction = np.argmax(self.__Q)
        self.__policy[bestAction] += 1 - ε
        self.__policyHistory.append(self.__policy.tolist())
     
    def showPolicyCurve(self):
        self.__policyHistory = np.array(self.__policyHistory)
        self.__policyHistory = self.__policyHistory.T
        plt.figure()
        iterations = range(0,np.shape(self.__policyHistory)[1])
        for i in range(0,actionNumber):
            plt.plot(iterations,self.__policyHistory[i],label = "action %d"%i)
        plt.legend(loc=0,ncol=1)
        plt.xlabel('iteration')   
        plt.savefig('策略的变化曲线-第%d个卖家.jpg'%self.__sellerIndex, dpi=300)
        # save figure time as provider j's strategy changing curve
        plt.show()
  
def qlearning(N,M,c,V,a,y_min,y_max,actionNumber,times):
    #Q表参数
    # Q table parameters
    df = 0.99   #discount factor,折扣因子。推荐：df ∈ [0.88,0.99]
    ε = 0.04     #探索概率 exploration probability
    
    #初始化N个卖家对象,放进列表allSellers里。
    # initialize N providers, and put all the devices in allSellers
    allSellers = []
    for j in range(0,N):
        tmpSeller = Seller(j,actionNumber,c[j],y_min,y_max)
        allSellers.append(tmpSeller)
 
    #使用Q learning算法更新Q表、策略
    # Q-learning update Q-table, action
    pricesHistory = []          #用于保存"所有的【卖家报价】"的历史记录 save the provider's histroy auxiliary price histroy
    purchasesHistory = []       #用于保存"所有的【单个买家的总购买数量】"的历史记录 save each device's total resource purchase history
    for t in range(0,times):
        #设置参数
        # parameter initialization
        α = 1 / (20 + t)    #用于更新Q值的学习率  update learning rate of Q value
        
        #每个卖家根据自己的策略选择动作，获得动作编号数组actions[].
        #Each provider chooses an action based on the learned polices, and code the action into actions list.
        actions = []
        for tmpSeller in allSellers:
            actions.append(tmpSeller.actionSelect())
        actions = np.array(actions)
        
        #把动作编号数组actions[]转换成由动作值y组成的数组ys[]。
        #transfor action numbers into y, and put all the ys in a list
        ys = action2y(actions,actionNumber,y_min,y_max)
        
        #保存本次迭代的【所有卖家的报价】
        # save all the providers price
        prices = 1 / ys
        pricesHistory.append(prices)
        
        #根据卖家的动作值y,换算出买家的购买数量
        # get the amount of resources purchased by device based on y
        X = []  #X是由数组x_j组成的数组 X is a list of x_i
        for j in range(0,N):
            x_j = V * ys[j] + a - np.e #x_j是由【所有买家向第j个卖家购买的产品数量】组成的数组
            # x_j is a list describes the amount of resources purchased by each device from provider j
            X.append(x_j)
        X = np.array(X)
        
        #保存本次迭代的【每个买家的总购买数量】
        #save [the total amount of resources purchased by each device], sum_X_ij of over N
        purchases = X.sum(axis = 0) #purchases是由【每个买家的总购买数量】组成的数组 purchases is the sum_X_ij of over N
        purchasesHistory.append(purchases)
    
        #更新Q表、平均策略、策略
        #update Q table, policy and average policy
        for j in range(0,N):
            allSellers[j].updateQ(actions,X[j],α,df) #更新Q表 update Q table
            allSellers[j].updatePolicy(ε)   #更新策略 update policy
    #Q learning算法结束。
    # Q-learning completes
    return pricesHistory,purchasesHistory,times
    
#    #需要展示的数据:
#    
#    #画出每个卖家的price的变化
#    plt.subplot(2,2,1)
#    ph.plotHistory(pricesHistory,times,None,"Price","provider")
#    #画出每个买家的总购买数量的变化
#    plt.subplot(2,2,3)
#    ph.plotHistory(purchasesHistory,times,"Iterations during Q-learning","Demand","device")
    