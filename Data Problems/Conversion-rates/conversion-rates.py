import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

#data preparation
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

print data_array[:10]











