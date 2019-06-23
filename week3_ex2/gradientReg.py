import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sigmoid import sigmoid


def gradientReg(theta, X, y,lam):
    m = len(y)
    grad = np.zeros([len(theta),1])
    hypothesis = sigmoid(np.dot(X,theta))
    error = hypothesis - y
    grad = (1/m)*(np.dot(X.T , error))
    grad[1:] = grad[1:] + (lam / m) * theta[1:]
    return grad