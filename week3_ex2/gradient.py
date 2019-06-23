import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sigmoid import sigmoid

def gradient(theta, X, y):
    m = len(y)
    hypothesis = sigmoid(np.dot(X,theta))
    error = hypothesis - y
    grad = (1/m)*(np.dot(X.T , error))
    return grad