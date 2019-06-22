import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros([iterations, 1])

    for i in range(iterations):
        err = np.dot(X, theta) - y
        err = np.dot(X.T, err)
        theta = theta - (alpha/m) * err
        J_history[i] = computeCostMulti(X,y,theta)
    
    print(J_history)
    return theta
