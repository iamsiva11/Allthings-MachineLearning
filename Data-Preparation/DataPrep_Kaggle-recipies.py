import pandas as pd
import numpy as np

#Pimas data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)

#Backup training data
train_old = dataframe.copy()

#Drop columns
dataframe = dataframe.drop(['skin',], axis=1)  
#dataframe.head(10)

#Are there any columns missing values?
dataframe.isnull().any()




