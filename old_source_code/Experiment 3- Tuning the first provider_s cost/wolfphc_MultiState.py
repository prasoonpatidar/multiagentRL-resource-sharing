#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
推荐： recommend
y ∈ (0.0200,0.0500]
"""

"""
备注：notes
在这个版本中，用于计算R的x_{i,j}可以取连续的数值，由下式计算得到。
In this version, x_{i,j} used for calculating R is continous, and calcluated by the following formula
x_j = V * yAll[j] + A / d - np.e

状态：状态设置为上一时刻的卖家的联合动作
State: all the providers price at t-1
"""

import numpy as np
import matplotlib.pyplot as plt 
import performance as pf
import time 
    
def sellerAction2y(sellerAction,sellerActionSize,y_min,y_max):
    y = y_min + (y_max - y_min) / sellerActionSize * sellerAction
    return y

def allSellerActions2stateIndex(allSellerActions,N,sellerActionSize):
    stateIndex = 0
    for i in range(0,N):
        stateIndex = stateIndex * sellerActionSize + allSellerActions[i]
    return stateIndex

class Seller:
    def __init__(self,sellerIndex,selleractionSize,stateSize,c_j,y_min,y_max):
        #卖家的个人信息 provider information
        self.__c = c_j   #该卖家的成本系数 cost for that provider
        
        #卖家的编号、状态空间大小、动作空间大小
        # provider index, state space, and action space
        self.__sellerIndex = sellerIndex
        self.__stateSize = stateSize
        self.__actionSize = selleractionSize
        self.__y_min = y_min
        self.__y_max = y_max
        
        
        #当前状态、当前动作
        # current state, current action
        self.__currentState = np.random.randint(0,self.__stateSize)
        self.__nextState = -1
        self.__action = -1
        self.__y = -1
        
        #Q表、平均策略、策略、计数器
        # Q table, mean policy, policy, count
        self.__Q = np.zeros((self.__stateSize,self.__actionSize))
        self.__policy = np.ones((self.__stateSize,self.__actionSize)) \
                        * (1 / self.__actionSize)
        self.__meanPolicy = np.ones((self.__stateSize,self.__actionSize)) \
                        * (1 / self.__actionSize)
        self.__count = np.zeros(self.__stateSize) 
        
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
#        print("self.__Q[self.__currentState] =",self.__Q[self.__currentState])
#        print("α =",α)
        
        #计算即时奖励R instant reward
        yAll = sellerAction2y(allSellerActions,sellerActionSize,
                              self.__y_min,self.__y_max)
        R = self.__y / sum(yAll) * (1 / self.__y - self.__c) * sum(x_j)
#        print("相应的R值 =",R)
        
        #拿到下一时刻的状态 get next state
        self.__nextState = allSellerActions2stateIndex(allSellerActions,\
                                                       N,sellerActionSize)
#        print("self.__nextState =",self.__nextState)
        
        #更新Q表 Q table update
#        print("self.Qmax() =",self.Qmax())
        self.__Q[self.__currentState][self.__action] = \
        (1 - α) * self.__Q[self.__currentState][self.__action] \
        + α * (R + df * self.Qmax())
#        print("Q表更新后：")
#        print("self.__Q[self.__currentState] =",self.__Q[self.__currentState])
        
    def updateMeanPolicy(self):
#        print("平均策略更新前：")
#        before mean policy updated
#        print("self.__count[self.__currentState] =",self.__count[self.__currentState])
#        print("self.__meanPolicy[self.__currentState] =\n",self.__meanPolicy[self.__currentState])
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
        self.__count[self.__currentState] += 1
        self.__meanPolicy[self.__currentState] += \
        (self.__policy[self.__currentState] - \
         self.__meanPolicy[self.__currentState]) \
         / self.__count[self.__currentState]
#        print("平均策略更新后：")
    #        after mean policy updated
#        print("self.__count[self.__currentState] =",self.__count[self.__currentState])
#        print("self.__meanPolicy[self.__currentState] =\n",self.__meanPolicy[self.__currentState])
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
        
    def updatePolicy(self,δ_win):
        #print("\n策略更新前：self.__policy =",self.__policy)
        # before policy updated
        δ_lose = 50 * δ_win
#        print("\n卖家",self.__sellerIndex,":") # provider
#        print("策略更新前：")
       # before policy updated
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
#        print("策略更新后：")
#        print("self.__Q[self.__currentState] =\n",self.__Q[self.__currentState])
#        print("self.__policy[self.__currentState] =\n",self.__policy[self.__currentState])
        
        
    def updateState(self):
#        print("状态更新前：")
#  before state updated
#        print("self.__currentState =",self.__currentState)
        self.__currentState = self.__nextState
#        print("状态更新后：")
    # after state updated
#        print("self.__currentState =",self.__currentState)
        
    def showPolicy(self):
        print("\n卖家",self.__sellerIndex,":") # provider
        print("self.__meanPolicy =\n",self.__meanPolicy)
        print("self.__policy =\n",self.__policy)
        
class Record:
    def __init__(self,N,length):
        self.__index = 0
        self.__N = N
        self.__length = length
        self.__arr = np.array([[-1] * self.__N] * self.__length)#self.__arr是np.array类型的
        # self.__arr is np array
    
    def isConverged(self,actions):
        self.__arr[self.__index] = actions#actions的类型是np.array
        # action is np array
        self.__index = (self.__index + 1) % self.__length
        variance = np.var(self.__arr,axis = 0)
        if sum(variance) == 0:
#            #画出arr
# plot arr
#            self.__arr = self.__arr.T
#            iterations = range(0,self.__length)
#            plt.figure()
#            for j in range(0,self.__N):
#                plt.plot(iterations,self.__arr[j],"o:",label = "seller %d"%(j + 1))
#            plt.legend()
#            plt.xlabel("the last %d iterations"%self.__length)
#            plt.ylabel("actions")
#            plt.savefig("收敛曲线.jpg",dpi = 300)
# covergency plot
#            plt.show()
            
            return True
        else:
            return False

        
     
def wolfphc_MultiState(N,M,c,V,a,y_min,y_max,actionNumber):
    #******（1）设置参数*************      paramter seting
    #Q表参数 Q table
    df = 0.30 #discount factor,折扣因子。推荐：df ∈ [0.88,0.99]
    # discount factor--recommend df ∈ [0.88,0.99]
    α = 1 / 3 #用于更新Q值的学习率 learning rate for updating Q table
    
    #卖家参数 provider paramters
    sellerActionSize = actionNumber  #卖家动作数 provider actions
    stateSize = sellerActionSize ** N    
    
    #******（2）初始化卖家们*************   initilize providers
    #初始化卖家 initilization
    sellers = []
    for j in range(0,N):
        tmpSeller = Seller(j,sellerActionSize,stateSize,c[j],y_min,y_max)
    #    tmpSeller.show()
        sellers.append(tmpSeller)
    
    #******（3）更新Q表、平均策略、策略*************
    # update Q table, mean policy and policy
    record = Record(N,500)#用于记录最近的连续500次的【所有卖家的动作的编号】#用于判断是否收敛
    # record the most recent 500 iterations [all the devices' index], for convergency checking
    start = time.perf_counter()
    timeLimit_min = 2 #timeLimit_min是以分钟为单位的限定时间 the unit of timeLimit_min is minutes
    t = -1
    while 1:
        #参数 parameters
        t += 1
        δ_win = 1 / (500 + 0.1 * t)
        
        #获得联结动作  actions get all the providers action
        allSellerActions = []
        for tmpSeller in sellers:
            allSellerActions.append(tmpSeller.actionSelect())
        allSellerActions = np.array(allSellerActions)
        yAll = sellerAction2y(allSellerActions,sellerActionSize,y_min,y_max)
    
        #更新Q表、平均策略、策略
        # update Q table, mean policy and policy
        #print("\n更新Q表、平均策略、策略:")
        #print update Q table, mean policy and policy
        for j in range(0,N):
            x_j = V * yAll[j] + a - np.e
            sellers[j].updateQ(allSellerActions,x_j,α,df,N,sellerActionSize)
            sellers[j].updateMeanPolicy()
            sellers[j].updatePolicy(δ_win)
            sellers[j].updateState()
        
        #判断是否已经收敛 check whether convergent
        #判断标准，如果在【最近的连续500次迭代】中，【所有卖家的报价】保持不变，则认为是已经收敛
        # check stardand, if all the providers' price remain the same in the most recent
        # 500 iterations, then we consider converges
        if record.isConverged(allSellerActions) == True:
            break
        #判断是否超出限定时间。如果超出，则返回False。
        # check convergence time, if over time,  return False
        stop = time.perf_counter()
        if (stop - start) / 60.0 > timeLimit_min:
            return False
    #Wolfphc算法结束  Wolfphc algorithm ends
    
    #******（4）返回 由【每个卖家的效益】组成的数组*************
    # return array of [each provider's utility]
    #拿到X[] get X[]
    X = []
    for j in range(0,N):
        x_j = V * yAll[j] + a - np.e #x_j是np.array x_j is np array
        X.append(x_j)
    X = np.array(X)
    #返回由【每个卖家的效益】组成的数组
    # return array of [each provider's utility]
    return pf.sellerRevenuesCalculator(X,yAll,N) - pf.sellerExpensesCalculator(X,yAll,c,N)
    
    
        
    
    #******（5）打印结果、画变化曲线************* 
#    print("\n打印policy:")     
#    for tmpSeller in sellers:
#        tmpSeller.showPolicy()
            
    #打印出所有卖家最后的单价p
#    P = Y[-1]
#    P = 1 / P
    #print("\n打印结果:")
    #print("P =",P)
    
    #画出Y的每一行，就是每一个卖家的y的变化
#    Y = np.array(Y)
#    Y = Y.T
#    iteration = range(0,np.shape(Y)[1])
#    plt.figure()
#    for j in range(0,N):
#        plt.plot(iteration,Y[j],label = "seller %d"%j)
#    plt.legend(loc=2,ncol=1)
#    plt.xlabel('iteration')  
#    plt.ylabel('y')  
#    plt.savefig('yWith%dSellersAnd%dBuyer.jpg'%(N,M), dpi=300)
#    plt.show()
    
    print("\nwolfphc_MultiSate:\nactions = %r"%allSellerActions)