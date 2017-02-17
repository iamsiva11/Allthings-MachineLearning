"""
Explore sklearn.preprocessing
"""
import numpy as numpy
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

X=[:, 0:8]
Y=[:, 8]


"""Normalise the data"""
from sklearn.preprocessing import Normalizer
scalar = Normalizer().fit(X)
NormalisedX =scalar.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(normalizedX[0:5,:])


"""binarization"""
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(binaryX[0:5,:])


"""Rescale data (between 0 and 1)"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


"""Label Encode """
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
#resultant encoded values
encoded_y  = encoder.transform(Y_Train)
#View all the clases that were encoded
encoder.clases_

"""Label encoder - decode (inverse_transform)"""

#Decoding the labels back again
encoder.inverse_transform(Y)