#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:20:19 2020

@author: xuhuiying
"""

#gradient decent
"""
现在✅：
当30个以内的卖家，100个以内的买家时的纳什均衡点：
find the nash equilibrium for providers (<30) and devices (<100)
p ∈ [0.0020,0.0040]
y ∈ [250,500]
"""

"""
调整参数：c V a的上下边界的值 
parameter tuning: change the upper and lower bound of C, V, a
c_max = 20
c_min = 10

V_max = 500
V_min = 200

a_max = 2
a_min = 1.5

目标： goal
p ∈ [0.01,0.02]
y ∈ [50,100]
"""

import numpy as np
import matplotlib.pyplot as plt
import performance as pf

#参数设置 input setting

N = 30 #卖家的个数 number of providers
M = 100 #买家的个数 number of devices
c = np.random.uniform(0.001,0.002,size=N) #卖家的单位CPU工作时的成本 required CPU occupied time for providers
V = np.random.uniform(0.01,0.02,size=M) #买家完成任务获得的奖励 device's reward for complete the task
a = np.random.uniform(2.3,2.7,size=M) #买家的任务的工作量（完成任务所需的CPU工作时） device's number of tasks
                                    # (total CPU time needed for processing the task)

#print("np.e =",np.e)
#print("c =",c)
#print("V =",V)
#print("a =",a)

#梯度下降法 gradient decent
μ = 10000 #梯度下降法的更新步长 step size updation for gradient decent
Y = np.ones(N) * 700.0 #矩阵，用来记录第τ步的所有定价的倒数，每个元素是y_{j,τ}
# matrix for record all the y (reverse of price) at step t, each element named as y_{j,τ}
tmp_y = Y.copy() #当前y current y

#print("μ =",μ)
#print("Y =",Y)
#print("tmp_y =",tmp_y)
#print("id(Y) = ",id(Y))
#print("id(tmp_y) = ",id(tmp_y),"\n")

for t in range(0,3000):
    for j in range(0,N):
        
        #根据当前的tmp_y求出y_j的梯度的数值 tmp_gradient
        # get the tmp_gradient based on tmp_y
        tmp_a = 0
        for i in range(0,M):
            tmp_a += - (a[i] - np.e) - V[i] * c[j] * (tmp_y[j] ** 2)
        tmp_b = sum(tmp_y) - tmp_y[j]
        tmp_c = 0
        for i in range(0,M):
            tmp_c += - 2 * V[i] * c[j] * tmp_y[j] + \
            (V[i] - a[i] * c[j] + np.e * c[j])
        tmp_d = sum(tmp_y)
        tmp_gradient = (tmp_a + tmp_b * tmp_c) / (tmp_d ** 2)
        
        #用梯度tmp_gradient更新tmp_y[j]
        # update tmp_y[j] based on tmp_gradient
        tmp_y[j] += μ * tmp_gradient
        #print("tmp_y =",tmp_y)
        #print("Y =",Y)
    #添加tmp_y到Y的最后一行 add tmp_y as the last row of Y
    Y = np.row_stack((Y,tmp_y))
    #print("Y =",Y)
    
#打印出矩阵Y来看看 print matrix
#print("\nY =\n",Y)
    
#把Y转置 reverse Y
Y = Y.T

last_y = Y[:,-1] #均衡点的定价p the price p at nash equilibrium
print("last_y =",last_y)
print("min(last_y) =",min(last_y),"\nmax(last_y) =",max(last_y))

#画出Y的每一行，就是每一个卖家的y的变化
#Plot the each row of Y, which is each provider's y changes
iteration = range(0,np.shape(Y)[1])
plt.figure()
for j in range(0,N):
    plt.plot(iteration,Y[j],label = "%d"%j)
plt.legend(loc=0,ncol=1)
plt.xlabel('iteration')  
plt.ylabel('y')  
plt.savefig('梯度下降-y.jpg', dpi=300)
plt.show()

#把Y转置 reverse Y
P = 1 / Y
#print("\nP =\n",P)
last_p = P[:,-1] #均衡点的定价p the price p at nash equilibrium
print("last_p =",last_p)
print("min(last_p) =",min(last_p),"\nmax(last_p) =",max(last_p))

#画出P的每一行，就是每一个卖家的p的变化
#Plot the each row of Y, which is each provider's y changes
#iteration = range(0,np.shape(P)[1])
plt.figure()
for j in range(0,N):
    plt.plot(iteration,P[j],label = "%d"%j)
plt.legend(loc=0,ncol=1)
plt.xlabel('iteration')  
plt.ylabel('p')  
plt.savefig('梯度下降-p.jpg', dpi=300) # gradient decent.jpg
plt.show()


#拿到X[] get X[]
X = []
for j in range(0,N):
    x_j = V * last_y[j] + a - np.e #x_j是np.array x_j is an np array
    X.append(x_j)
X = np.array(X)
print("X = \n",X)

#得到卖家效用值、买家的效用值 get providers and devices utilities
sellerUtilities,buyerUtilities = pf.sellerAndBuyerUtilities(X,last_y,c,V,a,N,M)
print("sellerUtilities =\n",sellerUtilities)
print("buyerUtilities =\n",buyerUtilities)

