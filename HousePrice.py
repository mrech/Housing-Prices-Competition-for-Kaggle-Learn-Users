
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, Binarizer, PowerTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# 0. IMPORT DATASETS

# Alternatuive: keep_default_na = False
data = pd.read_csv('train.csv')
dt_test = pd.read_csv('test.csv')

# variables
data.columns

# Devide input features from output feature
# Remove Id and response variable
X = data.iloc[::, 1:-1]
y = data.loc[:, 'SalePrice']

# Create a 20% test set to see if the model generalize well
X_train, test_X, y_train, test_y = train_test_split(X, y, test_size=0.2,
                                                    random_state=765)

# Apply the same transformation to the test set.
X_test = dt_test.iloc[::, 1:]

# return dtypes in data
X_train.dtypes

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


def summary_stats_numeric(X, y):
    '''
    Plot summary stats together with univariate dist and bivariate dist
    Input: numerical variable and response variable
    Output: plot and summary stats
    '''

    # Remove missing
    y = y[X.notnull()]
    X = X.dropna()

    # Create a figure instance, and the two subplots
    fig = plt.figure(figsize=(50, 100))
    fig.suptitle('Univariate distribution & descriptive statistics\n{}'.format(
        round(pd.DataFrame.describe(X), 2)))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(top=0.75)

    # Plot on ax1 with the ax argument
    sns.distplot(X, ax=ax1)
    sns.regplot(X, y, ax=ax2)

    plt.show()


def bivariate_distr_categorical(X, y):
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

X_train = X_train.fillna(value=values)
test_X = test_X.fillna(value=values)
X_test = X_test.fillna(value=values)

# Check for missing values on the output
# np.sum(pd.isna(y_train))

# summary_stats_numeric(y_train, y_train)

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
    X_train[i] = X_train[i].astype('category')

# Visual inspection
# for i in cat_list:
#   summary_stats_category(X_train[i])

# Note: MasVnrType, Electrical (have missing values)

# Adjust for ordered categorical variables
ord_list0 = ['OverallQual', 'OverallCond']

for i in ord_list0:
    X_train[i] = X_train[i].astype(pd.api.types.CategoricalDtype(ordered=True))

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X_train[i])

# 1st strategy
# Adjust for time Periods >> format time as category
# assign 0 to NaN

# Generalize
# Then we can adjust the zeros to the most frequent category (if necessary)
date_list = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']

imp_period_miss = SimpleImputer(missing_values=np.nan,
                                strategy='constant', fill_value=0)
imp_period_miss = imp_period_miss.fit(X_train[date_list])
X_train[date_list] = imp_period_miss.transform(X_train[date_list])
test_X[date_list] = imp_period_miss.transform(test_X[date_list])

# Impute the missing to the test set
X_test[date_list] = imp_period_miss.transform(X_test[date_list])

for i in date_list:
    X_train[i] = X_train[i].astype(pd.api.types.CategoricalDtype(ordered=True))

# Visual inspection
# for i in date_list:
#    summary_stats_category(X_train[i])

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
    X_train[i] = X_train[i].astype(order)

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X_train[i])

# Adjust for ordered categorical variables
order = pd.api.types.CategoricalDtype(['NoFeature',
                                       'No',
                                       'Mn',
                                       'Av',
                                       'Gd'], ordered=True)

ord_list2 = ['BsmtExposure']

for i in ord_list2:
    X_train[i] = X_train[i].astype(order)

# Visual inspection
# for i in ord_list:
#    summary_stats_category(X_train[i])

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
    X_train[i] = X_train[i].astype(order)

# Visual inspection
# for i in ord_list:
#   summary_stats_category(X_train[i])

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
    X_train[i] = X_train[i].astype(order)

# Visual inspection
#for i in ord_list4:
#   summary_stats_category(X_test[i])

# 2.2. imput missing

# Missing values for train data
col_miss = pd.DataFrame(np.sum(pd.isna(X_train), axis=0),
                        columns=['N_missing'])

col_miss = col_miss[col_miss['N_missing'] > 0]

# Missing values for test data
col_miss_test = pd.DataFrame(np.sum(pd.isna(X_test), axis=0),
                             columns=['N_missing'])

col_miss_test = col_miss_test[col_miss_test['N_missing'] > 0]

# DATA RELATIONSHIP INVESTIGATION
# (Knowing your data and the relationship between them)
# bivariate distribution with the predictor

# summary_stats_category(X_train['MasVnrType'])
# bivariate_distr_categorical(X_train['Electrical'])
# summary_stats_numeric(X_train['MasVnrArea'], y)
# summary_stats_category(X_train['Electrical'])

cat_var = X_train.select_dtypes(include=['category']).columns
num_var = X_train.select_dtypes(include=['int', 'float']).columns

# compute correlation matrix

corr = X_train.corr()

# Set up plt figure
fig, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap
sns.heatmap(corr, center=0, linewidths=.5)
plt.show()

# Adjust for COLLINEARITY/Multicollinearity
# compute the variance inflation factor
# Drop GarageCars because highly correlated with GarageCars
# and most of the Sales Price variability already explained by GrageArea
# same variables calculated with different measurements
# (alternatively keep it)
X_train = X_train.drop('GarageCars', axis=1)
test_X = test_X.drop('GarageCars', axis=1)
X_test = X_test.drop('GarageCars', axis=1)

# High correlation with TotBsmSF and FistFloorSF
# Instead of dropping the variable we add them together

X_train['TotalSF'] = X_train['TotalBsmtSF'] + \
    X_train['1stFlrSF'] + X_train['2ndFlrSF']
X_train = X_train.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)

test_X['TotalSF'] = test_X['TotalBsmtSF'] + \
    test_X['1stFlrSF'] + test_X['2ndFlrSF']
test_X = test_X.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)

X_test['TotalSF'] = X_test['TotalBsmtSF'] + \
    X_test['1stFlrSF'] + X_test['2ndFlrSF']
X_test = X_test.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)

num_var = num_var.drop(['GarageCars', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'])
num_var = num_var.insert(len(num_var), 'TotalSF')

# High negative correlation between BsmtFinSF1 and BsmtUnfSF
# check relationship within the variables and within the sale price.
X_train['BsmtFinSF'] = X_train['BsmtFinSF1'] + \
    X_train['BsmtFinSF2'] - X_train['BsmtUnfSF']
# BsmtFinSF2 almost irrelevant
test_X['BsmtFinSF'] = test_X['BsmtFinSF1'] + \
    test_X['BsmtFinSF2'] - test_X['BsmtUnfSF']
X_test['BsmtFinSF'] = X_test['BsmtFinSF1'] + \
    X_test['BsmtFinSF2'] - X_test['BsmtUnfSF']

#sns.jointplot(X_train['BsmtUnfSF'], X_train['BsmtSF'])
X_train = X_train.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
test_X = test_X.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
X_test = X_test.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)

num_var = num_var.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'])
num_var = num_var.insert(len(num_var), 'BsmtFinSF')

# FEATURES ENGINEERING
# Many bathrooms variables with few observations each add them all together
# General trend more bathroom better, and full is better than half
# represent it numerically
X_train['Bathroom'] = X_train['BsmtFullBath'] + X_train['FullBath'] + \
    0.5*X_train['BsmtHalfBath'] + 0.5*X_train['HalfBath']
test_X['Bathroom'] = test_X['BsmtFullBath'] + test_X['FullBath'] + \
    0.5*test_X['BsmtHalfBath'] + 0.5*test_X['HalfBath']
X_test['Bathroom'] = X_test['BsmtFullBath'] + X_test['FullBath'] + \
    0.5*X_test['BsmtHalfBath'] + 0.5*X_test['HalfBath']

#summary_stats_numeric(X_train['Bathroom'], y_train)

X_train = X_train.drop(
    ['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], axis=1)
test_X = test_X.drop(
    ['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], axis=1)
X_test = X_test.drop(
    ['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], axis=1)

num_var = num_var.drop(
    ['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'])
num_var = num_var.insert(len(num_var), 'Bathroom')

# Many porch variables with few observations each add them all together
# to reduce the number of missing/0
# summary_stats_numeric(X_train['OpenPorchSF'], y_train)
X_train['TotPorchSF'] = X_train['WoodDeckSF'] + X_train['OpenPorchSF'] +\
    X_train['EnclosedPorch'] + X_train['3SsnPorch'] + X_train['ScreenPorch']
test_X['TotPorchSF'] = test_X['WoodDeckSF'] + test_X['OpenPorchSF'] +\
    test_X['EnclosedPorch'] + test_X['3SsnPorch'] + test_X['ScreenPorch']
X_test['TotPorchSF'] = X_test['WoodDeckSF'] + X_test['OpenPorchSF'] +\
    X_test['EnclosedPorch'] + X_test['3SsnPorch'] + X_test['ScreenPorch']

X_train = X_train.drop(
    ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1)
test_X = test_X.drop(
    ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1)
X_test = X_test.drop(
    ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1)

num_var = num_var.drop(
    ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'])
num_var = num_var.insert(len(num_var), 'TotPorchSF')

# summary stats, categorical, numerical, and bivariate
# Transform pool in categorical variable (too many zeros)
#summary_stats_numeric(X_train['PoolArea'], y_train)
# summary_stats_category(X_train['PoolQC'])
#bivariate_distr_categorical(X_train['PoolQC'], y_train)
X_train['WithPool'] = X_train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
#bivariate_distr_categorical(X_train['WithPool'], y_train)
test_X['WithPool'] = test_X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
X_test['WithPool'] = X_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

X_train = X_train.drop(['PoolArea'], axis=1)
test_X = test_X.drop(['PoolArea'], axis=1)
X_test = X_test.drop(['PoolArea'], axis=1)

# Econding as binomial
num_var = num_var.drop(['PoolArea'])
cat_var = cat_var.insert(len(cat_var), 'WithPool')
bin_list = ['WithPool']

X_train['WithFireplace'] = X_train['Fireplaces'].apply(
    lambda x: 1 if x > 0 else 0)
#bivariate_distr_categorical(X_train['Fireplaces'], y_train)
test_X['WithFireplace'] = test_X['Fireplaces'].apply(
    lambda x: 1 if x > 0 else 0)
X_test['WithFireplace'] = X_test['Fireplaces'].apply(
    lambda x: 1 if x > 0 else 0)

# Encoding as binomial
num_var = num_var.drop(['Fireplaces'])
cat_var = cat_var.insert(len(cat_var), 'WithFireplace')
bin_list.append('WithFireplace')

X_train['WithAtMostOneKitchen'] = X_train['KitchenAbvGr'].apply(
    lambda x: 0 if x > 1 else 1)
test_X['WithAtMostOneKitchen'] = test_X['KitchenAbvGr'].apply(
    lambda x: 0 if x > 1 else 1)
X_test['WithAtMostOneKitchen'] = X_test['KitchenAbvGr'].apply(
    lambda x: 0 if x > 1 else 1)

# Encoding as binomial
num_var = num_var.drop(['KitchenAbvGr'])
cat_var = cat_var.insert(len(cat_var), 'WithAtMostOneKitchen')
bin_list.append('WithAtMostOneKitchen')

# drop variables with have most of the observations (>75%) equal to 0
X_train = X_train.drop(['LowQualFinSF', 'MiscVal'], axis=1)
test_X = test_X.drop(['LowQualFinSF', 'MiscVal'], axis=1)
X_test = X_test.drop(['LowQualFinSF', 'MiscVal'], axis=1)

num_var = num_var.drop(['LowQualFinSF', 'MiscVal'])


# check categorical variables for futures engineering!
#foo = pd.DataFrame(y_train)
#foo.index = X_train.index
#foo = pd.concat([X_train, y_train], axis=1)
#sns.catplot('Alley', 'SalePrice', hue='Street', data=foo)
# plt.show()

# drop utilities all are AllPub (no add info)
# summary_stats_category(X_train['Street'])
#bivariate_distr_categorical(X_train['Street'], y_train)
X_train = X_train.drop(['Utilities'], axis=1)
test_X = test_X.drop(['Utilities'], axis=1)
X_test = X_test.drop(['Utilities'], axis=1)

cat_list.remove('Utilities')
cat_var = cat_var.drop(['Utilities'])

# check for overlap of info within categorical variables
#sns.catplot('SaleCondition', 'SalePrice', hue='SaleType', data=foo)
# plt.show()

# Adjust yearbuild, year remodelled, garageyear >>> group them meaningful way
# timeserie analysis
#sns.swarmplot(pd.cut(X_train['YearRemodAdd'], 5), y_train)

#bivariate_distr_categorical(pd.cut(X_train['YearRemodAdd'], 5), y_train)
#plt.show()

#X_train['YearBuilt'], fitbins = pd.cut(X_train['YearBuilt'], 3,
                                    #labels=['Late800First900', 'Mind900', 'Late900First000'], retbins=True)

#X_test['YearBuilt'] = pd.cut(X_test['YearBuilt'], bins=fitbins, labels=['Late800First900', 'Mind900', 'Late900First000'])
#test_X['YearBuilt'] = pd.cut(test_X['YearBuilt'], bins=fitbins, labels=['Late800First900', 'Mind900', 'Late900First000'])

#cat_list.append('YearBuilt')
#date_list.remove('YearBuilt')

# YearRemodAdd' 

#X_train['YearRemodAdd'], fitbins = pd.cut(X_train['YearRemodAdd'], 5,
#                                    labels=['49-62', '62-74', '74-86', '86-98',  '98-10'], retbins=True)

#X_test['YearRemodAdd'] = pd.cut(X_test['YearRemodAdd'], bins=fitbins, labels=['49-62', '62-74', '74-86', '86-98',  '98-10'])
#test_X['YearRemodAdd'] = pd.cut(test_X['YearRemodAdd'], bins=fitbins, labels=['49-62', '62-74', '74-86', '86-98',  '98-10'])

#cat_list.append('YearRemodAdd')
#date_list.remove('YearRemodAdd')

#  GarageYrBlt'
X_train = X_train.drop(['GarageYrBlt'], axis = 1)
X_test = X_test.drop(['GarageYrBlt'], axis = 1)
test_X = test_X.drop(['GarageYrBlt'], axis = 1)

date_list.remove('GarageYrBlt')
cat_var = cat_var.drop(['GarageYrBlt'])
# scatterplot of two variable, regression line and 95% confidence
# Adjust for OUTLIERS and HIGH LEVERAGE POINTS !!
# Implement a better mechanism to spot them
# Drop observations with high leverage points in LotFrontage

# Find outliers/leverage points base on observation
# hue to check against categorical variables
# (avoid to delete important information)
#sns.pairplot(X_train[num_var].dropna(), height=1.3)
#plt.show()

# (laverage stats with multiple predictors)
# for i in num_var:
#   sns.regplot(X_train[i], y_train)
#   plt.show()

#summary_stats_numeric(X_train['BsmtFinSF'], y_train)
drop_rows = np.concatenate((
    list(X_train['TotalSF'].index[X_train['TotalSF'] > 6500]), # (already in the list)
    list(X_train['BsmtFinSF'].index[X_train['BsmtFinSF'] > 4000]), # (already in the list)
    list(X_train['LotFrontage'].index[X_train['LotFrontage'] > 300]),
    list(X_train['LotArea'].index[X_train['LotArea'] > 100000]),
    list(X_train['GrLivArea'].index[X_train['GrLivArea'] > 4000]))) # (already in the list)


def unique(list_value):
    unique = []

    for i in list_value:
        if i not in unique:
            unique.append(i)

    return unique


drop_rows = unique(drop_rows)

X_train = X_train.drop(drop_rows)
y_train = y_train.drop(drop_rows)

#sns.pairplot(X_train[num_var].dropna(), height=1.3)
#plt.show()
#for i in num_var:
    #summary_stats_numeric(X_train[i], y_train)

#summary_stats_numeric(X_train['GrLivArea'], y_train)

# Input top frequency category checking for price range
# NOTE: Imputation of missing needs to be the same in the test set
# Generalize for all categorical (test set - we can other missing categories)

imp_cat_miss = SimpleImputer(strategy='most_frequent')
imp_cat_miss = imp_cat_miss.fit(X_train[cat_var])
X_train[cat_var] = imp_cat_miss.transform(X_train[cat_var])
test_X[cat_var] = imp_cat_miss.transform(test_X[cat_var])

# Impute the missing to the test set
X_test[cat_var] = imp_cat_miss.transform(X_test[cat_var])

# Generalize for all numerical
# assign median value to missing in LotFrontage

imp_num_miss = SimpleImputer(missing_values=np.nan, strategy='median')
imp_num_miss = imp_num_miss.fit(X_train[num_var])

X_train[num_var] = imp_num_miss.transform(X_train[num_var])
test_X[num_var] = imp_num_miss.transform(test_X[num_var])

# Impute the missing to the test set
X_test[num_var] = imp_num_miss.transform(X_test[num_var])

#summary_stats_numeric(X_train['LotFrontage'], y_train)

## BOX-COX Transformation
# Let's relax the linearity assumption 
# https://stats.stackexchange.com/questions/18844/when-and-why-should-you-take-the-log-of-a-distribution-of-numbers
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html
# for negative values

# check numerical variable if non-linear transformation are needed
# or apply Box cox transformation for skewd numerical data
# (Apply after imputation)
# for i in num_var:
#    summary_stats_numeric(X_train[i], y_train)

# create a class
# add a constant to 0s in order to apply Box-Cox transformation
# find num_var with zeros
# find num_var with negative values


def ZeroNeg_num_var(col_list):
    '''
    Find num variables which contains zeros
    Input: list of numerica variable names
    '''
    zero_col = []
    neg_col = []

    for i in col_list:
        if any(X_train[i] < 0):
            neg_col.append(i)
        elif any(X_train[i] == 0):
            zero_col.append(i)

    return zero_col, neg_col


def min_value_zero(zero_col):
    '''
    Calculate min values above zero, divide them by 2 and
    return imputed values for each columns

    Input: list of name columns with zero values
    Return: imput value for each columns
    '''
    imput_val = []
    for i in zero_col:
        x = min(X_train[i].iloc[np.where(X_train[i] > 0)])
        imput_val.append(x/2)

    return imput_val


zero_col, neg_col = ZeroNeg_num_var(num_var)
imput_val = min_value_zero(zero_col)

# imput on the training and test sets
for num, txt in enumerate(zero_col):
    X_train[txt] = [x+imput_val[num] if x == 0 else x for x in X_train[txt]]
    test_X[txt] = [x+imput_val[num] if x == 0 else x for x in test_X[txt]]
    X_test[txt] = [x+imput_val[num] if x == 0 else x for x in X_test[txt]]

# transform training data & save lambda value
strictly_pos = num_var.drop(neg_col)
transf_col = []
fitted_lambda = []
for i in strictly_pos:
    transf, fitted = stats.boxcox(X_train[i])
    transf_col.append(transf)
    fitted_lambda.append(fitted)

transf_col = pd.DataFrame(transf_col).T
transf_col.columns = strictly_pos
transf_col.index = X_train.index
X_train[strictly_pos] = transf_col

# Use lambda value to transform test data
test_X[strictly_pos] = stats.boxcox(test_X[strictly_pos], fitted_lambda)
X_test[strictly_pos] = stats.boxcox(X_test[strictly_pos], fitted_lambda)

# Use power transform for neg_col
power_transf = PowerTransformer().fit(X_train[neg_col])

X_train[neg_col] = power_transf.transform(X_train[neg_col])
X_test[neg_col] = power_transf.transform(X_test[neg_col])
test_X[neg_col] = power_transf.transform(test_X[neg_col])

# investigate probplot
# for i in strictly_pos:
#    fig, ax = plt.subplots()
#    # plot against normal distribution
#    prob = stats.probplot(X_train[i], dist=stats.norm, plot=ax)
#    ax.set_title('Probplot against normal distribution %s' % (i))
#    plt.show()

# drop MasVnrArea many zeros values and
# do not follow the normality assumption
X_train = X_train.drop('MasVnrArea', axis=1)
test_X = test_X.drop('MasVnrArea', axis=1)
X_test = X_test.drop('MasVnrArea', axis=1)

num_var = num_var.drop(['MasVnrArea'])

# 2.3 Feature Encoding

# Encode ordinal features

ord_enc_0 = OrdinalEncoder(categories=[list(range(1, 11))]*len(ord_list0))
ord_enc_0 = ord_enc_0.fit(X_train[ord_list0])
ord_enc_0.categories_  # First category represent the baseline
X_train[ord_list0] = ord_enc_0.transform(X_train[ord_list0])
test_X[ord_list0] = ord_enc_0.transform(test_X[ord_list0])

# Encode the features on the test set
X_test[ord_list0] = ord_enc_0.transform(X_test[ord_list0])

ord_enc_1 = OrdinalEncoder(
    categories=[['NoFeature', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]*len(ord_list1))
ord_enc_1 = ord_enc_1.fit(X_train[ord_list1])
ord_enc_1.categories_  # Baseline (NoFeature)
X_train[ord_list1] = ord_enc_1.transform(X_train[ord_list1])
test_X[ord_list1] = ord_enc_1.transform(test_X[ord_list1])

# Encode the features on the test set
X_test[ord_list1] = ord_enc_1.transform(X_test[ord_list1])

ord_enc_2 = OrdinalEncoder(categories=[['NoFeature',
                                        'No',
                                        'Mn',
                                        'Av',
                                        'Gd']]*len(ord_list2))
ord_enc_2 = ord_enc_2.fit(X_train[ord_list2])
ord_enc_2.categories_  # baseline it is the first on the list
X_train[ord_list2] = ord_enc_2.transform(X_train[ord_list2])
test_X[ord_list2] = ord_enc_2.transform(test_X[ord_list2])

# Encode the features on the test set
X_test[ord_list2] = ord_enc_2.transform(X_test[ord_list2])

ord_enc_3 = OrdinalEncoder(categories=[['NoFeature',
                                        'Unf',
                                        'LwQ',
                                        'Rec',
                                        'BLQ',
                                        'ALQ',
                                        'GLQ']]*len(ord_list3))
ord_enc_3 = ord_enc_3.fit(X_train[ord_list3])
X_train[ord_list3] = ord_enc_3.transform(X_train[ord_list3])
test_X[ord_list3] = ord_enc_3.transform(test_X[ord_list3])

# Encode the features on the test set
X_test[ord_list3] = ord_enc_3.transform(X_test[ord_list3])

ord_enc_4 = OrdinalEncoder(categories=[['Sal',
                                        'Sev',
                                        'Maj2',
                                        'Maj1',
                                        'Mod',
                                        'Min2',
                                        'Min1',
                                        'Typ']]*len(ord_list4))
ord_enc_4 = ord_enc_4.fit(X_train[ord_list4])
X_train[ord_list4] = ord_enc_4.transform(X_train[ord_list4])
test_X[ord_list4] = ord_enc_4.transform(test_X[ord_list4])

# Encode the features on the test set 
#X_test.Functional.unique()
X_test[ord_list4] = ord_enc_4.transform(X_test[ord_list4])

cat_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_enc = cat_enc.fit(X_train[cat_list])
cat_enc.categories_

cat = pd.DataFrame(cat_enc.transform(X_train[cat_list]))
X_train = X_train.drop(cat_list, axis=1)

# Adjust for the random indexing
cat.index = X_train.index
X_train = pd.concat([X_train, cat], axis=1)

cat_test_X = pd.DataFrame(cat_enc.transform(test_X[cat_list]))
test_X = test_X.drop(cat_list, axis=1)
# Adjust for random indexing
cat_test_X.index = test_X.index
test_X = pd.concat([test_X, cat_test_X], axis=1)

# Encode the features on the test set
cat_test = pd.DataFrame(cat_enc.transform(X_test[cat_list]))
X_test = X_test.drop(cat_list, axis=1)
X_test = pd.concat([X_test, cat_test], axis=1)

# Binomial Encoding already done with if else statement
#bin_enc = Binarizer()
#bin_enc = bin_enc.fit(X_train[bin_list])
#bin_var = pd.DataFrame(bin_enc.transform(X_train[bin_list]))

# 2.4 Features Standardization
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# 1st strategy, take into account for sparsity and ourliers
# Extract variables with sparse data where at least 25% of observations is 0
descriptive = pd.DataFrame.describe(X_train[num_var])

# Standardize variable with very small standard deviations
small_std = descriptive.loc['std', :] < 5
small_std_var = small_std[np.where(small_std)[0]].index.tolist()

small_std_transf = MaxAbsScaler().fit(X_train[small_std_var])
X_train[small_std_var] = small_std_transf.transform(X_train[small_std_var])
test_X[small_std_var] = small_std_transf.transform(test_X[small_std_var])

# Standardize the features on the test set
X_test[small_std_var] = small_std_transf.transform(X_test[small_std_var])

# Standardize variables with outliers (Skewed distribution)
# Transformation is still effected by outliers try Box Cox
num_var = [i for i in num_var if i not in small_std_var]
descriptive = pd.DataFrame.describe(X_train[num_var])

skewed_transf = RobustScaler().fit(X_train[num_var])
X_train[num_var] = skewed_transf.transform(X_train[num_var])

# for i in num_var:
#    summary_stats_numeric(X_train[i], y_train)

test_X[num_var] = skewed_transf.transform(test_X[num_var])

# Standardize the features on the test set
X_test[num_var] = skewed_transf.transform(X_test[num_var])

# Standardize date
descriptive = pd.DataFrame.describe(X_train)
descriptive = descriptive.loc[['mean', 'std'], :].transpose()
descriptive.sort_values(by=['mean', 'std'])

# for i in date_list:
#    summary_stats_numeric(X_train[i],y_train)
# 0 imput to GarageYrBlt (feature engineer to create meaningfull variable and handle multicoll.)

yr_list = date_list[:-1]

yr_stand_transf = StandardScaler().fit(X_train[yr_list])
X_train[yr_list] = yr_stand_transf.transform(X_train[yr_list])
test_X[yr_list] = yr_stand_transf.transform(test_X[yr_list])

# Standardize the features on the test set
X_test[yr_list] = yr_stand_transf.transform(X_test[yr_list])

# Maintaining the Cyclic Representations of Month Sold
# https://datascience.stackexchange.com/a/24003

## Explanation / Intuition
#foo = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
# Circle Circumference = (2*pi*r) devide by 12 periods
# for each period we calculate sin and cos
# With this transformation any cyclical feature will be doubled.
#bar1 = np.cos((foo-1) * (2*np.pi/12))
#bar2 = np.sin((foo-1) * (2*np.pi/12))
#fig, ax = plt.subplots()
#plt.scatter(bar1, bar2)
# for i, txt in enumerate(foo-1):
#    ax.annotate(txt, (bar1[i], bar2[i]))
# plt.show()

MoSold_cos = np.cos((X_train[date_list[-1]]-1) * (2*np.pi/12))
MoSold_sin = np.sin((X_train[date_list[-1]]-1) * (2*np.pi/12))

#fig, ax = plt.subplots()
#plt.scatter(MoSold_cos, MoSold_sin)
# for i, txt in enumerate((X_train[date_list[-1]]-1)):
#    ax.annotate(txt, (MoSold_cos[i], MoSold_sin[i]))
# plt.show()

X_train = X_train.drop(date_list[-1], axis=1)
X_train['MoSold_cos'] = MoSold_cos
X_train['MoSold_sin'] = MoSold_sin

MoSold_cos = np.cos((test_X[date_list[-1]]-1) * (2*np.pi/12))
MoSold_sin = np.sin((test_X[date_list[-1]]-1) * (2*np.pi/12))
test_X = test_X.drop(date_list[-1], axis=1)
test_X['MoSold_cos'] = MoSold_cos
test_X['MoSold_sin'] = MoSold_sin

# Standardize the features on the test set
MoSold_cos = np.cos((X_test[date_list[-1]]-1) * (2*np.pi/12))
MoSold_sin = np.sin((X_test[date_list[-1]]-1) * (2*np.pi/12))
X_test = X_test.drop(date_list[-1], axis=1)
X_test['MoSold_cos'] = MoSold_cos
X_test['MoSold_sin'] = MoSold_sin

# 3. MODELS

# log-transform the target variable, since submission are evaluated
# on logarithm value of sales price.

y_train = np.log(y_train)
test_y = np.log(test_y)

# 3.1 Linear Regression with multiple variables
linear_reg = LinearRegression()

# 4. PREDICT AND ACCURACIES
# Cross Validation Strategy
# https://scikit-learn.org/stable/model_selection.html#model-selection
# K-fold cross validation with K = 5 or 10 provides a good compromise for this
# bias-variance tradeoff.

# Choice of Metric
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted
# value and the logarithm of the observed sales price.
# (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)


def cv_rmse(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=234)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kf,
                                    scoring='neg_mean_squared_error'))
    return rmse


# Check results on the training set
reg = LinearRegression().fit(X_train, y_train)
theta = reg.coef_
train_prediction = np.dot(X_train, theta) + reg.intercept_
rmse_train = np.sqrt(
    np.sum(np.power(y_train-train_prediction, 2))/len(y_train))
print('rmse_train for linear regression: ', rmse_train)

# After droping leverege points in LotFrontage rmse remain the same 0.0938

# Check results on the test set
test_prediction = pd.Series(np.dot(test_X, theta) + reg.intercept_)
test_prediction.index = test_X.index

rmse_test = np.sqrt(np.sum(np.power(test_y-test_prediction, 2))/len(test_y))
print('rmse_test for linear regression: ', rmse_test)
# 204505871.74563393
# after 1839666592.5665817
# after adjusting for collinearity (Garage and TotSF) 990527540.2560297
# after yearbuilt rmse_test for linear regression:  378469691.54623526
# after yearRemodelled rmse_test for linear regression:  1541625486.464488


# Check results on the cross validation
score = cv_rmse(linear_reg)
print("\nLinearRegression score: {:.4f} ({:.4f})\n".format(
    score.mean(), score.std()))
print('Before drpping LotFrontage LinearRegression score: 5056093844.8706 (4064033453.8808)')
print('After dropping LotFrontage LinearRegression score: 1047559176.9406 (620718954.2598)')
print('After adjusting for GarageCars TotalSF and score: LinearRegression score: 0.1443 (0.0194)')
print('BsmtSF LinearRegression score: 0.1334 (0.0173)')
print('Box-Cox transformation LinearRegression score: 1299196641.8624 (1523032431.9894)')
print('After yearBuild groupping LinearRegression score: 591093739.0732 (363526804.1697)')

# Error Analysis on the test data
# analysis of residuals: OVERFITTING
def residuals_vs_fitted(observed_y, fitted_y):
    residuals = observed_y - fitted_y
    sns.residplot(fitted_y, residuals, lowess=True,
               line_kws={'color': 'red'})
    plt.title('Fitted vs Residuals')
    plt.xlabel('Fitted values (TestPrediction)')
    plt.ylabel('Residuals')
    plt.show()

# We have large leverage observations (creates clusters).
# implement models robust to outliers and leverage points

# 3.2 Lasso Regression
# Find the optimal alpha
alpha_list = [0.00015625, 0.0003125, 0.0004125,
              0.0005125, 0.0006125, 0.000625, 0.00125]

# Narrow step between min points
#np.arange(0.0003125, 0.000625, 0.0001)

# Create a set of models with different degrees or any other variants

score = []
for elem in alpha_list:
    lasso = Lasso(alpha=elem, random_state=56)
    score.append(cv_rmse(lasso).mean())

# Regularization and Bias/Variance graph

def tuning_parameter(alpha, score):
    plt.plot(alpha, score)
    plt.title('Regularization Bias/Variance')
    plt.xlabel('alpha')
    plt.ylabel('Cross validation error')
    plt.scatter(alpha[np.argmin(score)],
                score[np.argmin(score)], c='r')
    plt.text(alpha[np.argmin(score)],
             score[np.argmin(score)], s='Min: {}'.format(alpha_list[np.argmin(score)]))
    plt.show()

#tuning_parameter(alpha_list, score)

# Select the best combo that produces the lowest error on cv
lasso = Lasso(alpha_list[np.argmin(score)], random_state=756)
score = cv_rmse(lasso)
print("\nLasso cv score: {:.4f} ({:.4f})\n".format(
    score.mean(), score.std()))
print('BsmtSF Lasso score: 0.1133 (0.0125)')
print('yearBuilt groupping Lasso score: 0.1147 (0.0127)')

# Check relevant Lasso coefs. 
lasso_fit = lasso.fit(X_train, y_train)
lasso_test_prediction = lasso_fit.predict(test_X)

rmse_test = np.sqrt(
    np.sum(np.power(test_y-lasso_test_prediction, 2))/len(test_y))
print('rmse test with lasso regression: ', rmse_test)

#residuals_vs_fitted(test_y, lasso_test_prediction)
# sign of slight non-linearity and leverage/ouliers 

# 3.3 Rigid Regression
# Find the optimal alpha
alpha_list = [0.6 , 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7 ,
       0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8 ]

score = []
for elem in alpha_list:
    ridge = Ridge(alpha=elem, random_state=56)
    score.append(cv_rmse(ridge).mean())

#tuning_parameter(alpha_list, score)

# Select the best combo that produces the lowest error on cv
ridge = Ridge(alpha_list[np.argmin(score)], random_state=756)
score = cv_rmse(ridge)
print("\n Ridge cv score: {:.4f} ({:.4f})\n".format(
    score.mean(), score.std()))

# Check relevant Lasso coefs. 
ridge_fit = ridge.fit(X_train, y_train)
ridge_test_prediction = ridge_fit.predict(test_X)

rmse_test = np.sqrt(
    np.sum(np.power(test_y-ridge_test_prediction, 2))/len(test_y))
print('rmse test with ridge regression: ', rmse_test)

residuals_vs_fitted(test_y, ridge_test_prediction)





## TO DO
# create git branch wiht the different feature engineer and transformations
# in order to select the best performer models.