import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

def displayData(data , x,y):
    img = 0
    fig , subplt = plt.subplots(10,10,figsize=(10,10))
    for i in range(10):
        for j in range(10):
            subplt[i,j].imshow(data[img].reshape((x,y), order = 'F'))
            subplt[i,j].axis('off')
            img+=1
    plt.show()
    return 1