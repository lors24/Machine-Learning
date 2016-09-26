# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:54:39 2016

@author: loredp
"""
import functions as f
from loadFittingDataP2 import getData
#import numpy.linalg as npl
#import matplotlib.pyplot as plt
#import numpy as np
#import math

data = getData(ifPlotData=True)

X = data[0][:,None]
Y = data[1][:,None]
M = 2
l = 0.5

res = f.eval(X,Y,M,l=l,plot = True)
print "The SSE is " + str(res[1])
print "Weight vectors" + str(res[0])

