# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:29:48 2016

@author: loredp
"""

from loadFittingDataP2 import getData
import numpy as np
import matplotlib.pyplot as plt
import math
import functions as f

data = getData(ifPlotData=False)

X = data[0][:,None]
Y = data[1][:,None] 
     
w = f.ml_weight(X,Y,10)    

#x = np.arange(0,1,.02)

#y_pred = np.zeros(len(x))
#for i in range(len(theta)):
#    y_pred += x**i*theta[i]
#    
#y_real = np.cos(x*math.pi)+np.cos(x*2*math.pi)

#plt.plot(X,Y,'o')
#plt.plot(x,y_real)
#plt.plot(x,y_pred)
#plt.xlabel('x')
#plt.show()
#plt.ylabel('y')

# SSE





