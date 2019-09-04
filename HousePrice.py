
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Lasso


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
                                                    random_state=42)

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
#np.sum(pd.isna(y_train))

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
#    summary_stats_category(X_train[i])
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
# for i in ord_list:
#    summary_stats_category(X_train[i])

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

# scatterplot of two variable, regression line and 95% confidence
# Adjust for OUTLIERS and HIGH LEVERAGE POINTS
# (laverage stats with multiple predictors)
# for i in num_var:
#    sns.regplot(X_train[i], y)
#    plt.show()

# compute correlation matrix

corr = X_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True

# Set up plt figure
fig, ax = plt.subplots(figsize=(50, 100))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask
# Adjust for COLLINEARITY/Multicollinearity
# compute the variance inflation factor
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=.5)
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.show()

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

summary_stats_numeric(X_train['LotFrontage'], y_train)

# 2.3 Feature Encoding

# Encode ordinal features
ord_enc_0 = OrdinalEncoder()
ord_enc_0 = ord_enc_0.fit(X_train[ord_list0])
ord_enc_0.categories_  # 0 category represent the baseline (1)
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
X_test[ord_list4] = ord_enc_4.transform(X_test[ord_list4])

cat_enc = OneHotEncoder(sparse = False, handle_unknown='ignore')
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
test_X = pd.concat([test_X, cat_test_X], axis = 1)

# Encode the features on the test set
cat_test = pd.DataFrame(cat_enc.transform(X_test[cat_list]))
X_test = X_test.drop(cat_list, axis=1)
X_test = pd.concat([X_test, cat_test], axis = 1)

# 2.4 Features Standardization
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# 1st strategy, take into account for sparsity and ourliers
# Extract variables with sparse data where at least 25% of observations is 0
descriptive = pd.DataFrame.describe(X_train[num_var])
sparse = descriptive.loc['25%', :] == 0
sparse_var = sparse[np.where(sparse)[0]].index.tolist()

sparse_transf = MaxAbsScaler().fit(X_train[sparse_var])
X_train[sparse_var] = sparse_transf.transform(X_train[sparse_var])
test_X[sparse_var] = sparse_transf.transform(test_X[sparse_var])

# Standardize the features on the test set
X_test[sparse_var] = sparse_transf.transform(X_test[sparse_var])

# for i in sparse_var:
#    summary_stats_numeric(X_train[i], y_train)

num_var = num_var.tolist()

# list comprehension
num_var = [i for i in num_var if i not in sparse_var]
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
num_var = [i for i in num_var if i not in small_std_var]

skewed_transf = RobustScaler().fit(X_train[num_var])
X_train[num_var] = skewed_transf.transform(X_train[num_var])
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
# summary_stats_numeric(y_train,y_train)

# Linear Regression with multiple variables
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
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kf,
                                    scoring='neg_mean_squared_error'))
    return rmse

# Check results on the training set
reg = LinearRegression().fit(X_train, y_train)
theta = reg.coef_
prediction = np.dot(X_train, theta) + reg.intercept_
rmse1 = np.sqrt(np.sum(np.power(y_train-prediction, 2))/len(y_train))

# overfitting
score = cv_rmse(linear_reg)
print("\nLinearRegression score: {:.4f} ({:.4f})\n".format(
    score.mean(), score.std()))

# Find the optimal alpha
alpha_list = [0.00015625, 0.0003125, 0.0004125,
              0.0005125, 0.0006125, 0.000625, 0.00125]

# Narrow step between min points
#np.arange(0.0003125, 0.000625, 0.0001)

# Create a set of models with different degrees or any other variants

score = []
for elem in alpha_list:
    lasso = Lasso(alpha=elem, random_state=1)
    score.append(cv_rmse(lasso).mean())

# Regularization and Bias/Variance graph
plt.plot(alpha_list, score)
plt.title('Regularization Bias/Variance')
plt.xlabel('alpha')
plt.ylabel('Cross validation error')

plt.scatter(alpha_list[np.argmin(score)],
            score[np.argmin(score)], c='r')
plt.text(alpha_list[np.argmin(score)],
         score[np.argmin(score)], s='Min: {}'.format(alpha_list[np.argmin(score)]))
plt.show()

# Select the best combo that produces the lowest error on cv
lasso = Lasso(alpha_list[np.argmin(score)], random_state=1)
score = cv_rmse(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(
    score.mean(), score.std()))

# TO DO

# check cross-validation implementation
# implement a function or class that transform cyclic representation
# interpret the coefs.

# 2.3.1 Log transformation
# https://stats.stackexchange.com/questions/18844/when-and-why-should-you-take-the-log-of-a-distribution-of-numbers
# FEATURE ENGINEERING
# Group by category - create summary variables (ex. garage) to solve for MULTICOLLINEARITY
# boxplot by categories
# X_train['MoYrSold'] = X_train['MoSold'].map(str) + '-' + X_train['YrSold'].map(str)
# X_train['MoYrSold'] = pd.to_datetime(X_train['MoYrSold'], format='%m-%Y')
# PCA and Correlation graph
# log transformation
