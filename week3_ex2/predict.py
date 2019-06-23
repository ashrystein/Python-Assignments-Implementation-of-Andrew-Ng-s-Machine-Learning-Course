import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunction import costFunction
from gradient import gradient
from fmintnc import fmintnc
from sigmoid import sigmoid

def predict(theta, X , m):
    p = np.zeros([m,1])
    hypothesis = sigmoid(np.dot(X,theta))
    for i in range(m):
        if (hypothesis[i][0]) >= 0.5:
            p[i][0] = 1
    return p