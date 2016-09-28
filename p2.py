# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:29:48 2016

@author: loredp
"""

from loadFittingDataP2 import getData
import numpy as np
import matplotlib.pyplot as plt
import math
import functions as fun

data = getData(ifPlotData=True)

X = data[0][:,None]
Y = data[1][:,None] 
     
M = 2
w = fun.ml_weight(X,Y,M)   


fun.evaluate(X,Y,M,w, plot = True, f = fun.q2)

#Check gradient

#w2, res = fun.evaluate(X,Y,8,basis = fun.cos, plot = True, f = fun.q2)

#P3

M = 2
l = 0
wr = fun.ridge(X,Y,M,l) 
sse = fun.SSE(X,Y,wr,M,l)
s = sse(wr)
fun.evaluate(X,Y,M,wr, l, plot = True, f = fun.q2)

