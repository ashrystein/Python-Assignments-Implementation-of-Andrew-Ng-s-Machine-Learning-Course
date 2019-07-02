import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt


def polyFeatures(X, p):
    m = X.shape[0]
    Xpoly = np.zeros((m,p))
    n = Xpoly.shape[1]
    
    for i in range(n):
        Xpoly[:,i] = np.power(X[:,1],i+1)
    #print('>' , Xpoly.shape , '\n')
    return Xpoly