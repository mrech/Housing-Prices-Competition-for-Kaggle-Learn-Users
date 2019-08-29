
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# 0. Import dataset
#
# Alternatuive: keep_default_na = False
data = pd.read_csv('train.csv')

# variables
data.columns

# Devide input features from output feature
# Remuve Id and response variable
X = data.iloc[::, 1:-1]
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
        round(pd.DataFrame.describe(X), 2)))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(top=0.75)

    # Plot on ax1 with the ax argument
    sns.boxplot(X, ax=ax1)
    sns.distplot(X, ax=ax2)

    plt.show()


def bivariate_distr_categorical(X):
    '''
    Plot relationship beween the categorical variable and the target variable
    Input: categorical variable 
    '''
    sns.boxplot(X, y)
    plt.show()


# 2. DATA PREPROCESSING

# 2.0 Assign the right data type and correct import error

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
ord_list0 = ['OverallQual', 'OverallCond']

for i in ord_list0:
    X[i] = X[i].astype(pd.api.types.CategoricalDtype(ordered=True))

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X[i])

# 1st strategy
# Adjust for time Periods >> format time as category
# assign 0 to NaN

# Generalize
# Then we can adjust the zeros to the most frequent category (if necessary)
date_list = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']

imp_period_miss = SimpleImputer(missing_values=np.nan,
                                strategy='constant', fill_value=0)
imp_period_miss = imp_period_miss.fit(X[date_list])
X[date_list] = imp_period_miss.transform(X[date_list])


for i in date_list:
    X[i] = X[i].astype(pd.api.types.CategoricalDtype(ordered=True))

# Visual inspection
# for i in date_list:
#    summary_stats_category(X[i])

# Adjust for ordered categoricMasVnrTypeal variables
order = pd.api.types.CategoricalDtype(['NoFeature',
                                       'Po',
                                       'Fa',
                                       'TA',
                                       'Gd',
                                       'Ex'], ordered=True)

ord_list1 = ['BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
             'GarageCond', 'PoolQC']

for i in ord_list1:
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

ord_list2 = ['BsmtExposure']

for i in ord_list2:
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

ord_list3 = ['BsmtFinType1', 'BsmtFinType2']

for i in ord_list3:
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

ord_list4 = ['Functional']

for i in ord_list4:
    X[i] = X[i].astype(order)

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X[i])

# 2.2. imput missing

col_miss = pd.DataFrame(np.sum(pd.isna(X), axis=0),
                        columns=['N_missing'])

col_miss = col_miss[col_miss['N_missing'] > 0]

# DATA RELATIONSHIP INVESTIGATION
# (Knowing your data and the relationship between them)
# bivariate distribution with the predictor

# summary_stats_category(X['MasVnrType'])
# bivariate_distr_categorical(X['Electrical'])
# summary_stats_numeric(X['MasVnrArea'])
# summary_stats_category(X['Electrical'])

cat_var = X.select_dtypes(include=['category']).columns
num_var = X.select_dtypes(include=['int', 'float']).columns

# scatterplot of two variable, regression line and 95% confidence
# Adjust for OUTLIERS
# for i in num_var:
#    sns.regplot(X[i], y)
#    plt.show()

# compute correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True

# Set up plt figure
fig, ax = plt.subplots(figsize=(50, 100))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask
# Adjust for COLLINEARITY
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.5)
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.show()

# Input top frequency category checking for price range
# NOTE: Imputation of missing needs to be the same in the test set
# Generalize for all categorical (test set - we can other missing categories)

imp_cat_miss = SimpleImputer(strategy='most_frequent')
imp_cat_miss = imp_cat_miss.fit(X[cat_var])
X[cat_var] = imp_cat_miss.transform(X[cat_var])

summary_stats_numeric(X['LotFrontage'])

# Generalize for all numerical
# assign median value to missing in LotFrontage

num_var = X.select_dtypes(include=['int', 'float']).columns
imp_num_miss = SimpleImputer(missing_values=np.nan, strategy='median')
imp_num_miss = imp_num_miss.fit(X[num_var])

X[num_var] = imp_num_miss.transform(X[num_var])

# 2.3 Feature Encoding

# Encode ordinal features
ord_enc_0 = OrdinalEncoder()
ord_enc_0 = ord_enc_0.fit(X[ord_list0])
ord_enc_0.categories_  # 0 category represent the baseline (1)
X[ord_list0] = ord_enc_0.transform(X[ord_list0])

ord_enc_1 = OrdinalEncoder(
    categories=[['NoFeature', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]*len(ord_list1))
ord_enc_1 = ord_enc_1.fit(X[ord_list1])
ord_enc_1.categories_ # Baseline (NoFeature)
X[ord_list1] = ord_enc_1.transform(X[ord_list1])

ord_enc_2 = OrdinalEncoder(categories = [['NoFeature',
                                       'No',
                                       'Mn',
                                       'Av',
                                       'Gd']]*len(ord_list2))
ord_enc_2 = ord_enc_2.fit(X[ord_list2])
ord_enc_2.categories_ # baseline it is the first on the list
X[ord_list2] = ord_enc_2.transform(X[ord_list2])

ord_enc_3 = OrdinalEncoder(categories = [['NoFeature',
                                       'Unf',
                                       'LwQ',
                                       'Rec',
                                       'BLQ',
                                       'ALQ',
                                       'GLQ']]*len(ord_list3))
ord_enc_3 = ord_enc_3.fit(X[ord_list3])
X[ord_list3] = ord_enc_3.transform(X[ord_list3])

ord_enc_4 = OrdinalEncoder(categories = [['Sal',
                                       'Sev',
                                       'Maj2',
                                       'Maj1',
                                       'Mod',
                                       'Min2',
                                       'Min1',
                                       'Typ']]*len(ord_list4))
ord_enc_4 = ord_enc_4.fit(X[ord_list4])
X[ord_list4] = ord_enc_4.transform(X[ord_list4])

cat_enc = OneHotEncoder()
cat_enc = cat_enc.fit(X[cat_list])
cat_enc.categories_
cat = cat_enc.transform(X[cat_list]).toarray()

X = X.drop(cat_list, axis = 1)

X = pd.DataFrame.join(X, pd.DataFrame(cat))

# 2.4 Feature Normalization
# sin cos ciclic feature Month (MoSold) date_list[-1]
# 1st strategy
# date_list[:-1] normalize
# https://datascience.stackexchange.com/a/24003


# TO DO
# 2.3.1 Log transformation
# https://stats.stackexchange.com/questions/18844/when-and-why-should-you-take-the-log-of-a-distribution-of-numbers
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
