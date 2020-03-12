# import dependencies
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# read the data
filename = 'titanic.xls'
columns = ['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'ticket',
         'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest']

df = pd.read_excel(filename)
df = pd.DataFrame(df, columns=columns)

# split the data
train, test = train_test_split(df, test_size=0.2)

train = train.drop('cabin', axis=1)

# fill missing values and convert all features into int

# for age
data = [train, test]
for data_set in data:
    mean = train['age'].mean()
    std = test['age'].std()
    is_null = data_set['age'].isnull().sum()
    # compute random numbers
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # fill missing values with mean
    age_slice = data_set['age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data_set['age'] = age_slice
    # not sure what this does
    # data_set['age'] = train['age'].astype(int)

# filling missing values for embarked and fare
common_value = 'S'
data = [train, test]
for data_set in data:
    data_set['embarked'] = data_set['embarked'].fillna(common_value)
    data_set['fare'] = data_set['fare'].fillna(0)
    data_set['fare'] = data_set['fare'].astype(int)

# Numeric conversion for sex and embarked
embarked = { 'S': 0, 'C': 1, 'Q': 2 }
genders = { 'male': 0, 'female': 1 }
data = [train, test]
for data_set in data:
    data_set['sex'] = data_set['sex'].map(genders)
    data_set['embarked'] = data_set['embarked'].map(embarked)



# use linear regression on it



# predict



# plot on graph



# calculate accuracy






# https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
