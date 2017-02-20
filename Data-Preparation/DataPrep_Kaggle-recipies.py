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

#Dealing with missing data
df.isnull().sum()


# Eliminating samples or features with missing values¶
#removing records containing missign values
removed_na_rows = df.dropna()
#removing columns containing missign values
removed_na_cols = df.dropna(axis=1)
#Only drop rows where all columns are NaN
removed_na_rows_if_all =df.dropna(how='all')
# drop rows that have not at least 4 non-NaN values
removed_na_rows_4max = df.dropna(thresh=4)
# only drop rows where NaN appear in specific columns (here: 'C')
removed_na_rows_forcolX = df.dropna(subset=['preg'])


# Imputing missing values
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)


# Handling categorical data
# Encode labels (Mapping ordinal features)
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print y

original_lables = class_le.inverse_transform(y)
print original_lables


pd.get_dummies(df[['price', 'color', 'size']])