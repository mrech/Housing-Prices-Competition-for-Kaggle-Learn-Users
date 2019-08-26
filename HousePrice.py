
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

# Import train dataset
data = pd.read_csv('train.csv')

# variables
data.columns

# Devide input features from output feature
X = data.iloc[::, :-1]
y = data.loc[:, 'SalePrice']

# return dtypes in data
X.dtypes


# 1. DATA EXPLORATION

# 1.1 Descriptive Statistics

# create a function which plot summary stats and distribution

def summary_stats_category(X):
    '''
    Plot summary stats together with univariate distribution.
    Input: categorical variable 
    Output: plot and summary stats
    '''

    # count frequency for each class
    freq = dict(X.value_counts())
    fig = plt.figure()
    fig.suptitle('Univariate distribution & descriptive statistics',
                 fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('{}'.format(pd.DataFrame.describe(X)))
    ax.set_ylabel('Frequency')
    ax.bar(list(freq.keys()), list(freq.values()))

    plt.show()


def summary_stats_numeric(X):
    '''
    Plot summary stats together with univariate distribution.
    Input: numerical variable
    Output: plot and summary stats
    '''

    # Remove missing
    X = X.dropna()
    # X['LotFrontage']
    #

    # Create a figure instance, and the two subplots
    fig = plt.figure()
    fig.suptitle('Univariate distribution & descriptive statistics\n{}'.format(
        round(pd.DataFrame.describe(X),2)))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(top=0.75)

    # Plot on ax1 with the ax argument
    sns.boxplot(X, ax=ax1)
    sns.distplot(X, ax=ax2)

    plt.show()

# 2. DATA PREPROCESSING

# 2.1 Features Encoding

# For the categories >> substitute all NA with NoFeature
# instead of missing it becomes 'No feature'

values = {'Alley': 'NoFeature', 'BsmtQual': 'NoFeature',
          'BsmtCond': 'NoFeature', 'BsmtExposure': 'NoFeature',
          'BsmtFinType1': 'NoFeature', 'BsmtFinType2': 'NoFeature',
          'FireplaceQu': 'NoFeature',
          'GarageType': 'NoFeature', 'GarageFinish': 'NoFeature',
          'GarageQual': 'NoFeature', 'GarageCond': 'NoFeature',
          'PoolQC': 'NoFeature', 'Fence': 'NoFeature',
          'MiscFeature': 'NoFeature'}

X = X.fillna(value=values)

# Check for missing values on the output
np.sum(pd.isna(y))

# Encode categorical variables
# check if values are included

cat_list = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
            'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
            'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
            'CentralAir', 'Electrical', 'GarageType', 'GarageFinish',
            'PavedDrive', 'Fence', 'MiscFeature', 'SaleType',
            'SaleCondition']

for i in cat_list:
    X[i] = X[i].astype('category')

# Visual inspection
# for i in cat_list:
#    summary_stats_category(X[i])
# Note: MasVnrType, Electrical (have missing values)

# Adjust for ordered categorical variables
ord_list = ['OverallQual', 'OverallCond']

for i in ord_list:
    X[i] = X[i].astype(pd.api.types.CategoricalDtype(ordered=True))

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X[i])

# Adjust for Periods >> format time as category
# assign 0 to NaN
X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0).astype(int)
year_list = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']

for i in year_list:
    X[i] = X[i].astype(pd.api.types.CategoricalDtype(ordered=True))

# Visual inspection
# for i in year_list:
#    summary_stats_category(X[i])

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['NoFeature',
                                       'Po',
                                       'Fa',
                                       'TA',
                                       'Gd',
                                       'Ex'], ordered=True)

ord_list = ['BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond',
            'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
            'GarageCond', 'PoolQC']

for i in ord_list:
    X[i] = X[i].astype(order)

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X[i])

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['NoFeature',
                                       'No',
                                       'Mn',
                                       'Av',
                                       'Gd'], ordered=True)

ord_list = ['BsmtExposure']

for i in ord_list:
    X[i] = X[i].astype(order)

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X[i])

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['NoFeature',
                                       'Unf',
                                       'LwQ',
                                       'Rec',
                                       'BLQ',
                                       'ALQ',
                                       'GLQ'], ordered=True)

ord_list = ['BsmtFinType1', 'BsmtFinType2']

for i in ord_list:
    X[i] = X[i].astype(order)

# Visual inspection
# for i in ord_list:
#   summary_stats_category(X[i])

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['Sal',
                                       'Sev',
                                       'Maj2',
                                       'Maj1',
                                       'Mod',
                                       'Min2',
                                       'Min1',
                                       'Typ'], ordered=True)

ord_list = ['Functional']

for i in ord_list:
    X[i] = X[i].astype(order)

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X[i])

# 2.2 Handle missing values

col_miss = pd.DataFrame(np.sum(pd.isna(X), axis=0),
                        columns=['N_missing'])

col_miss = col_miss[col_miss['N_missing'] > 0]

# TO DO 
# 1. imput missing 
# https://scikit-learn.org/stable/modules/impute.html#impute


# DATA RELATIONSHIP INVESTIGATION 
# (Knowing your data and the relationship between them)
# bivariate distribution with the predictor

# correlation and PCA


# 2.3 Feature Normalization

# 2.3.1 Log transformation
# https://stats.stackexchange.com/questions/18844/when-and-why-should-you-take-the-log-of-a-distribution-of-numbers

sns.distplot(np.log(y))
plt.show()


# TO DO
# FEATURE ENGINEERING
# boxplot by categories
#X['MoYrSold'] = X['MoSold'].map(str) + '-' + X['YrSold'].map(str)
#X['MoYrSold'] = pd.to_datetime(X['MoYrSold'], format='%m-%Y')

# Divide trainset into Train (60%), Cross Validation (20%), Test Set (20%)
# https://scikit-learn.org/stable/model_selection.html#model-selection
# K-fold cross validation with K = 5 or 10 provides a good compromise for this
# bias-variance tradeoff.
# Use ShuffleSplit good alternative to KFold
# PCA and Correlation graph
# log transformation
