import pandas as pd
import numpy as np

#dataset1 - pima-indians-diabetes
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names= ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset= pd.read_csv(url, names=names)

#Convert Types
dataset = dataset.astype(float)
print dataset.dtypes

#Delete a column
dataset.drop('test', axis=1, inplace=True)
print dataset.shape
print dataset.head(20)

#dataset2 - breast-cancer-wisconsin
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['Code', 'Clump-Thickness', 'Cell-Size', 'Cell-Shape', 'Adhesion', 'Single-Cell-Size', 'Bare-Nuclei', 'Chromatin', 'Nucleoli', 'Mitoses', 'Class']
dataset = pandas.read_csv(url, names=names)

# Handling Nan Values
# Mark value as NaN
print(pandas.unique(dataset['Bare-Nuclei']))
dataset[['Bare-Nuclei']] = dataset[['Bare-Nuclei']].replace('?', numpy.NaN)
print(pandas.unique(dataset['Bare-Nuclei']))

# Delete rows with NaN values
dataset[['Bare-Nuclei']] = dataset[['Bare-Nuclei']].replace('?', numpy.NaN)
dataset.dropna(axis=0, how='any', inplace=True)
print(dataset.shape)