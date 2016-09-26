from loadFittingDataP2 import getData
import numpy.linalg as npl
import numpy as np

data = getData(ifPlotData=True)

def mlw(X,Y,M):   
    return npl.inverse(np.matrix.tranpose(X)*X)*np.matrix.tranpose(X)
