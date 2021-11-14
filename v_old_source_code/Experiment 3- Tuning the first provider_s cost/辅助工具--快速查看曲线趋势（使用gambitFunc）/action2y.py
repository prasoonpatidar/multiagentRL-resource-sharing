#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:53:55 2020

@author: xuhuiying
"""

def action2y(action,actionNumber,y_min,y_max):
    y = y_min + (y_max - y_min) / actionNumber * action
    return y