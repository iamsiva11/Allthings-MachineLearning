"""
Diabetes Dataset
"""
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

def data_exploration():
    pass

def data_split(data, target):
    X, y = data, target
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=33) 
    return X_train, X_test, y_train, y_test


def model(X_train, y_train, X_test):
    #Modelling
    #lr_model = LinearRegression().fit(diabetes.data, diabetes.target)
#   lr_model = LinearRegression().fit(X_train,y_train)
    lr_model = SVR().fit(X_train,y_train)
    #expected = diabetes.target
    #predicted= lr_model.predict(diabetes.data)
    y_predicted= lr_model.predict(X_test)
    return model , y_predicted


#Model Evaluation
def model_eval(y_test, y_predicted):
    print "Mean squared error = %0.3f" % mse(y_test, y_predicted)
    print "R2 score = %0.3f" % r2_score(y_test, y_predicted)

if __name__=="__main__":
    
    #Load data
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = data_split(diabetes.data, diabetes.target)
    model, y_predicted = model(X_train, y_train, X_test)
    model_eval(y_test, y_predicted)

