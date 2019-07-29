import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def computeCostMulti(X, y, theta):
    m = len(y)
    J = 0
    hypothesis = np.dot(X,theta)
    Errors = (hypothesis - y)
    J = np.sum(np.power(Errors, 2)) / (2*m)
    return J


def computeCostMulti2(X, y, theta):
    m = len(y)
    J = 0
    hypothesis = np.dot(X,theta)
    Errors = (hypothesis - y)
    J = (np.dot(Errors.T,Errors)) / (2*m)
    return J[0][0]