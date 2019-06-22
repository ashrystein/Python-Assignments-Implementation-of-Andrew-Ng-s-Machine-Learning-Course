import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def featureNormalize(X):
    X = (X - np.mean(X))/np.std(X)
    return X

