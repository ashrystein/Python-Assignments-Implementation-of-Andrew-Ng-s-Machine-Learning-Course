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
from featuresCleaning import featuresCleaning
from computeCostMulti import computeCostMulti
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import xgboost as xgb



dataTrain = pd.read_csv('train.csv')
dataTrain , dropCols= featuresCleaning(dataTrain)
X = dataTrain.iloc[:,:-1].values
y = dataTrain['SalePrice'].values
y = y[:,np.newaxis]


bst = xgb.XGBRegressor(objective = 'reg:squarederror',colsample_bytree=0.4,gamma=0,learning_rate=0.001,max_depth=3,min_child_weight=1.5,n_estimators=10000,reg_alpha=0.75,reg_lambda=0.45,subsample=0.6,seed=42)
bst.fit(X,y)
pred2 = bst.predict(X)
pred2 = pred2[:,np.newaxis]


print ('Predicted Stock Index Price using XGBoost: \n' , pred2)

print ('root_mean_squared_error: \n' , sqrt(mean_squared_error(pred2,y)))
print ('XGBRegressor Score: \n' ,bst.score(X,y))

###########################################################

dataTest = pd.read_csv('test.csv')
dataTest = dataTest.drop(dropCols,1)
dataTest['GrLivArea'] = np.log(dataTest['GrLivArea'])
dataTest = dataTest.fillna(dataTest.mode().iloc[0])

le = preprocessing.LabelEncoder()
for col in dataTest:
    if dataTest[col].dtype == 'object':
        le = le.fit(dataTest[col])
        dataTest[col] = le.transform(dataTest[col])


Xtest = dataTest.iloc[:,:].values

pred = bst.predict(Xtest)
pred = pred[:,np.newaxis]
print ('Predicted Stock Index Price: \n' , pred)

dataSub = pd.read_csv('sample_submission.csv')
ySub = dataSub['SalePrice'].values
ySub = ySub[:,np.newaxis]
print ('Actuall root_mean_squared_error: \n' , sqrt(mean_squared_error(pred,np.log(ySub))))


sales = dataTest.iloc[:, 0]
label = pd.DataFrame(np.exp(pred))
result = pd.concat([sales, label], axis=1)
result.columns = ['Id', 'SalePrice']
result.to_csv('submission.csv', index=False)