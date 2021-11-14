#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#在python2的环境下执行

import numpy as np
import gambit
from fractions import Fraction

def action2y(action,actionNumber,y_min,y_max):#把动作的编号转换成对应的动作值y
    y = y_min + (y_max - y_min) / actionNumber * action
    return y  

def profile2actions(profile,actionNumber,N):#把list(nashProfile._profile)转换成由动作编号组成的数组
    #profile是一个列表，元素个数是actionNumber * N个。
    actions = np.array([0] * N)#actions是np.array类型
    for j in range(0,N):
        for action in range(0,actionNumber):
            index = j * actionNumber + action
            if profile[index] == 1:
                actions[j] = action
                break
    return actions

def gambitSolveGame(N,M,c,V,a,y_min,y_max,actionNumber):
    g = gambit.Game.new_table([actionNumber] * N)
    
    #给R表赋值
    for profile in g.contingencies:
        #profile 是list的类型。np.array(profile) 是np.array的类型
        ys = action2y(np.array(profile),actionNumber,y_min,y_max) #ys是是np.array的类型
        for j in range(0,N):
            x_j = V * ys[j] + a - np.e  #x_j是np.array的类型
            g[profile][j] = Fraction.from_float((ys[j] / sum(ys)) * \
             ( 1 / ys[j] - c[j])* sum(x_j))
#            print("g%r[%d] = %f"%(profile,j,g[profile][j]))
            
            
    #求解nash均衡点
    solver = gambit.nash.ExternalEnumPureSolver()
    solution = solver.solve(g)

#    向【标准输出】输出纳什均衡。输出的格式是由【每个卖家的动作编号】组成的数组
#    print("solution = %r\n"%solution)
    for nashProfile in solution:
        profile = list(nashProfile._profile)
        actions = profile2actions(profile,actionNumber,N)
        print("%s"%actions)
#        print("actions = %r"%actions)
    
#    print("N = %r"%N)
#    print("M = %r"%M)
#    print("c = %r"%c)
#    print("V = %r"%V)
#    print("a = %r"%a)
#    print("y_min = %r"%y_min)
#    print("y_max = %r"%y_max)
#    print("actionNumber = %r"%actionNumber)
    
    
