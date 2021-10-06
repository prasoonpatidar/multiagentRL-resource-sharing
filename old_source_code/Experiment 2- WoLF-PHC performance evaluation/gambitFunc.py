#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:59:57 2020

@author: xuhuiying
"""

#在python3的环境下执行
# run in python3

import numpy as np
import subprocess
import action2y
import performance as pf

def str2array(stdoutData):
    result = stdoutData.split("\n")
    result.remove('')
    #不论gambitSolveGame()找到几个nash均衡点,在【stdoutData.split("\n")返回的list】中，只有最后一个元素是‘’
    # regardless of the number of Nash Equilibrium Point we found in gambitSolveGame()
    # we return the last element in [stdoutData.split("\n")]
    if len(result) > 1:
        return False
    
    #len(result) == 1的情况
    # scenario, len(result) == 1
    result = result[0]
    result = result.split("[")[1]
    result = result.split("]")[0]
    result = result.split() #默认按空格分割 the default divider is Space
    actions = []
    for ele in result:
        actions.append(int(ele))
    actions = np.array(actions)
    return actions

def gambitFunc(N,M,c,V,a,y_min,y_max,actionNumber):
    str = "import numpy as np; import gambitSolveGame; gambitSolveGame.gambitSolveGame("+\
    "%r,%r,"%(N,M) + "np." + "%r,"%c + "np." + "%r,"%V + "np." + "%r,"%a \
          + "%r,%r,%r"%(y_min,y_max,actionNumber) + ")"
    commandLine = "python -c '" + str + "'\n"
    
    s = subprocess.Popen(commandLine,
                     shell = True,
                     env = {"CONDA_DEFAULT_ENV":"python2",
                            "CONDA_SHLVL":"2",
                            "CONDA_PREFIX":"/Applications/anaconda3/envs/python2",
                            "CONDA_PROMPT_MODIFIER":"(python2)",
                            "CONDA_PREFIX_1":"/Applications/anaconda3",
                            "PATH":"/Applications/anaconda3/envs/python2/bin:/Applications/anaconda3/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                            },
                     stdout = subprocess.PIPE,stderr = subprocess.PIPE)
    stdoutData,stderrData = s.communicate()
    stdoutData = bytes.decode(stdoutData)
    stderrData = bytes.decode(stderrData)
    
    #处理stdoutData，得到actions数组
    # process stdoutData, and get actions array
    actions = str2array(stdoutData)
    #actions可能是False（bool类型）,或者是由【所有卖家的动作的编号】组成的数组(np.array类型)
    #action data type can be bool or np array [all the providers index]
    if type(actions) == type(False):
        return False
    #得到ys数组
    #get np array ys
    ys = action2y.action2y(actions,actionNumber,y_min,y_max)#ys是np.array类型
    #得到X[]数组
    #get X[] np array
    X = []
    for j in range(0,N):
        x_j = V * ys[j] + a - np.e #x_j是np.array类型
        X.append(x_j)
    X = np.array(X)
    #返回七类性能指标
    # return the seven performance indicators
    return pf.performance(X,ys,c,V,a,N,M)
    
    
    
    
#    print("\ngambit:\n%s"%stdoutData)
#    print("stderrData =",stderrData)
    
#"PATH":os.environ["PATH"] + ":/usr/local/bin",