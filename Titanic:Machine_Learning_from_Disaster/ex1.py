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
from featuresCleaning import featuresCleaning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from Kfold import run_kfold



dataTrain = pd.read_csv('train.csv')
Xt= dataTrain.iloc[:,2:]
yt = dataTrain.iloc[:,1].values
yt = yt[:,np.newaxis]


Xt = featuresCleaning(Xt)

C =100
kernel ="rbf"
classifier = SVC(C = C,kernel = kernel , gamma = 30)
classifier.fit(Xt,yt.ravel())

print("Training Accuracy SVM:",(classifier.score(Xt,yt.ravel()))*100,"%")



clf = RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(Xt, yt.ravel())
clf = grid_obj.best_estimator_
clf.fit(Xt, yt.ravel())
predictions = clf.predict(Xt)
print("Training Accuracy RandomForestClassifier:",(accuracy_score(predictions,yt.ravel()))*100,"%")


dataTest =  pd.read_csv('test.csv')
Xtest= dataTest.iloc[:,1:]

Xtest = featuresCleaning(Xtest)

predictionsTest = clf.predict(Xtest)
ytest = classifier.predict(Xtest)

genderSubmissionData = pd.read_csv('gender_submission.csv')
actualYtest = genderSubmissionData.iloc[:,1]


clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)
clf2.fit(Xt, yt.ravel())
NNpredictions = clf2.predict(Xt)
NNytest = clf2.predict(Xtest)
print("Training Accuracy NN:",(accuracy_score(NNpredictions,yt.ravel()))*100,"%")



mask = (actualYtest == NNytest)
m = len(mask)
print('Actuall Accuracy:  ',accuracy_score(NNytest,actualYtest.ravel())*100 , '%')

passenger = dataTest.iloc[:, 0]
label = pd.DataFrame(NNytest)
result = pd.concat([passenger, label], axis=1)
result.columns = ['PassengerId', 'Survived']
result.to_csv('submission.csv', index=False)


print('Kfold >','\n')
run_kfold(clf2 , Xt , yt.ravel())