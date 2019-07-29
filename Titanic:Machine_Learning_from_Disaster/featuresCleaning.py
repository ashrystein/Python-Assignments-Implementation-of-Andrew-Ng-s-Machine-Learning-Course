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



def featuresCleaning(Xt):
    drop_columns = ['Cabin', 'Name','Ticket','Embarked']
    Xt.drop(drop_columns, axis=1, inplace=True)
    Xt['Age'] = Xt['Age'].fillna((Xt['Age'].mean()))
    Xt['Fare'] = Xt['Fare'].fillna((Xt['Fare'].mean()))
    Xt['SibSp'] = Xt['SibSp'].fillna((Xt['SibSp'].mean()))
    Xt['Parch'] = Xt['Parch'].fillna((Xt['Parch'].mean()))

    le = preprocessing.LabelEncoder()
    le = le.fit(Xt['Sex'])
    Xt['Sex'] = le.transform(Xt['Sex'])
    Xt = Xt.values

    return Xt