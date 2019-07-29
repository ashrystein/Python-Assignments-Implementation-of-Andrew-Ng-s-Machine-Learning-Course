import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imag
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


def featuresCleaning(dataTrain):
    total = dataTrain.isnull().sum().sort_values(ascending=False)
    percent = (dataTrain.isnull().sum()/dataTrain.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    dropCols = missing_data[missing_data['Total'] > 1].index
    dataTrain = dataTrain.drop((missing_data[missing_data['Total'] > 1]).index,1)
    dataTrain = dataTrain.drop(dataTrain.loc[dataTrain['Electrical'].isnull()].index)
    
    dataTrain.sort_values(by = 'GrLivArea', ascending = False)[:2]
    dataTrain = dataTrain.drop(dataTrain[dataTrain['Id'] == 1299].index)
    dataTrain = dataTrain.drop(dataTrain[dataTrain['Id'] == 524].index)


    if 'SalePrice' in dataTrain:
        dataTrain['SalePrice'] = np.log(dataTrain['SalePrice']) #invest
    dataTrain['GrLivArea'] = np.log(dataTrain['GrLivArea']) #invest

    
    le = preprocessing.LabelEncoder()
    for col in dataTrain:
        if dataTrain[col].dtype == 'object':
            le = le.fit(dataTrain[col])
            dataTrain[col] = le.transform(dataTrain[col])
    
    return dataTrain , dropCols