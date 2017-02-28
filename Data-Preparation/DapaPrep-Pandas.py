import pandas as pd

# Drop Columns
################
train_data = train_data.drop(['COLUMNNAME'], axis = 1)

# Apply function
################

# transform data to log scale
train['COLUMNNAME'] = train['COLUMNNAME'].apply(lambda x: np.log1p(x))

data.datetime = data.datetime.apply(pd.to_datetime)