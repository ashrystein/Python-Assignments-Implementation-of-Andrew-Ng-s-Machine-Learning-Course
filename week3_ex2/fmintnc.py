import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from costFunction import costFunction
from gradient import gradient
from sigmoid import sigmoid


def fmintnc(theta, X, y):
    theta_optimized = opt.fmin_tnc(func = costFunction, 
                 x0 = theta.flatten(),fprime = gradient, 
                 args = (X, y.flatten()))
    return theta_optimized[0]


