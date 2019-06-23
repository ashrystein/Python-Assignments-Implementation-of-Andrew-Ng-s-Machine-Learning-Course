import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g