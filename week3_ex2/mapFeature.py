import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(len(X1))[:,np.newaxis]
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),np.power(X2, j))[:,np.newaxis]))
    return out