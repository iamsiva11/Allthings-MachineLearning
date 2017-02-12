import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#Load Data
url = "https://raw.githubusercontent.com/David-Loughnane/conversionRates/master/conversion_data.csv"
names=["country","age","new_user","source","total_pages_visited","converted"]
dataset = pd.read_csv(url)
#dataset = pd.read_csv(url , names=names)

"""
#Mini EDA
print dataset.head(10)
print dataset[['age']].head(10)

for n in names:
    print dataset[n].unique()

#one hot encoding
#data preparation
data_array = dataset.values()
print data_array[:10]


"""    

"""
Data preparation
"""
data_array = dataset.values
print data_array[:10,:]

#Encode country
y = data_array[:, 0]
encoder = LabelEncoder()
encoder.fit(y)
encoded_y  = encoder.transform(y)
data_array[:, 0] = encoded_y 

#Encode source
y = data_array[:, 3]
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
data_array[:, 3] = encoded_y

data_array[:, 0] = map(int ,data_array[:, 0] )
data_array[:, 3] = map(int ,data_array[:, 3] )
#print data_array[:10]

X = data_array[:, :5]
y = data_array[:, 5]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33) 

y_train = np.array(y_train).astype(bool)
y_test = np.array(y_test).astype(bool)


"""
Modelling Process
"""

#Fit the model
lr = LogisticRegression()
lr.fit(X,Y)
y_pred = lr.predict(X_test)

#Evaluation, accuracy
print accuracy_score(y_test, y_pred)