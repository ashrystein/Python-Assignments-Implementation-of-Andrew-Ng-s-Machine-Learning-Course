import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sigmoid import sigmoid

def predictNN(Theta1, Theta2, X):
    m = X.shape[0]
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))

    z2 =(a1 @ Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((ones, a2))

    z3 = (a2 @ Theta2.T)
    h = sigmoid(z3)

    pred = np.argmax(h, axis = 1)
    pred = [e if e else 10 for e in pred]
    
    return pred