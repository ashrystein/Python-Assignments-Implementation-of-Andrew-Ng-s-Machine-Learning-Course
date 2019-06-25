import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sigmoid import sigmoid

def costFunctionReg(theta, X, y,lam):
    m = len(y)
    hypothesis = sigmoid(np.dot(X,theta))
    J = (-1/m) * np.sum(np.multiply(y , np.log(hypothesis)) + np.multiply((1 - y) , np.log(1 - hypothesis)))
    reg = (lam/(2*m)) * (np.sum(np.power(theta,2)))
    J += reg
    return J