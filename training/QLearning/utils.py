'''
Utility fuctions for Qlearning
'''

import math
import numpy as np
from scipy.optimize import minimize

# transform the action index to action y
def action2y(action,actionNumber,y_min,y_max):#把动作的编号转换成对应的动作值y
    y = y_min + (y_max - y_min) / actionNumber * action
    return y



