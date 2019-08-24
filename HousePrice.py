
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit

# Import train dataset
data = pd.read_csv('train.csv')

# 1. DATA PREPROCESSING

# 1.1 Features Encoding

# return dtypes in data
data.dtypes

# Adjust for categorical variables

cat_list = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
            'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
            'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
            'CentralAir', 'Electrical', 'GarageType', 'GarageFinish',
            'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',
            'SaleCondition']

for i in cat_list:
    data[i] = data[i].astype('category')

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['Very Excellent',
                                       'Excellent',
                                       'Very Good',
                                       'Good',
                                       'Above Average',
                                       'Average',
                                       'Below Average',
                                       'Fair',
                                       'Poor',
                                       'Very Poor'], ordered=True)

ord_list = ['OverallQual', 'OverallCond']

for i in ord_list:
    data[i] = data[i].astype(order)

# Adjust for date
year_list = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

for i in year_list:
    data[i] = pd.to_datetime(data[i], format='%Y')

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['Excellent',
                                       'Good',
                                       'Average/Typical',
                                       'Fair',
                                       'Poor'], ordered=True)

ord_list = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual',
            'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for i in ord_list:
    data[i] = data[i].astype(order)

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['Ex',
                                       'Gd',
                                       'TA',
                                       'Fa',
                                       'Po',
                                       'NA'], ordered=True)

ord_list = ['BsmtQual', 'BsmtCond']

for i in ord_list:
    data[i] = data[i].astype(order)

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['Gd',
                                       'Av',
                                       'Mn',
                                       'No',
                                       'NA'], ordered=True)

ord_list = ['BsmtExposure']

for i in ord_list:
    data[i] = data[i].astype(order)

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['GLQ',
                                       'ALQ',
                                       'BLQ',
                                       'Rec',
                                       'LwQ',
                                       'Unf',
                                       'NA'], ordered=True)

ord_list = ['BsmtFinType1', 'BsmtFinType2']

for i in ord_list:
    data[i] = data[i].astype(order)

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['Typ',
                                       'Min1',
                                       'Min2',
                                       'Mod',
                                       'Maj1',
                                       'Maj2',
                                       'Sev',
                                       'Sal'], ordered=True)

ord_list = ['Functional']

for i in ord_list:
    data[i] = data[i].astype(order)

# Month and Year sold
data['MoYrSold'] = data['MoSold'].map(str) + '-' + data['YrSold'].map(str)
data['MoYrSold'] = pd.to_datetime(data['MoYrSold'], format='%m-%Y')
data = data.drop(columns=['MoSold', 'YrSold'])

# 1.2 Handle missing values

sum(pd.isna(data['MSSubClass']))
pd.DataFrame.describe(data['MSSubClass'])
freq = data['MSSubClass'].value_counts()

data['MSZoning'] = data['MSZoning'].astype('category')

# TO DO
# Divide trainset into Train (60%), Cross Validation (20%), Test Set (20%)
# https://scikit-learn.org/stable/model_selection.html#model-selection
# K-fold cross validation with K = 5 or 10 provides a good compromise for this
# bias-variance tradeoff.
# Use ShuffleSplit good alternative to KFold
# PCA and Correlation graph
# log transformation