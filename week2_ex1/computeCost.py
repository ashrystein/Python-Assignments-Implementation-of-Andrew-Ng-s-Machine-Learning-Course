import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = len(y)
    J = 0
    hypothesis = np.dot(X,theta)
    Errors = (hypothesis - y)
    J = np.sum(np.power(Errors, 2)) / (2*m)
    return J
