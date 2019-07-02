import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def costFunctionReg(theta, X, y,lam):
    m = len(y)
    thetaZero = theta
    #thetaZero[0] = 0; 
    hypothesis = np.dot(X,theta)
    Errors = (hypothesis - y)
    J = np.sum(np.power(Errors, 2)) / (2*m)
    reg = (lam/(2*m)) * (np.sum(np.power(thetaZero,2)))
    J += reg
    return J