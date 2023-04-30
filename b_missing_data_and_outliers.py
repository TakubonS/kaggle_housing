import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')

# of missing data for each column
total = df_train.isnull().sum().sort_values(ascending=False)
# percentage of missing data for each column
# isnull().sum() sum of the # of isnull
# isnull().count() total # of rows
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

# for missing data over 15%, delete the column
# GarageX: GarageType, GarageFinish, GarageQual, GarageCond have high correlation with GarageCars and GarageArea
# so delete them too
# Same with BsmtX
# MasVnrX: MasVnrType, MasVnrArea have high correlation with OverallQual
# for electrical, only one missing, delete the single row, keep the variable

# overall delete everything that has missing data but electrical

# drop columns
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
# drop row for electrical
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# check if any missing data left
print("after: ", df_train.isnull().sum().max())


# outliers
# First, standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])

# top 10 and bottom 10, not outliers, but be careful if too far from 0
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print(low_range)
print(high_range)

# plot some
var = "GrLivArea"
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.savefig('find_outlier/GrLivArea_before.png')
# Find 2 outlier points, delete them

# print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2])
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.savefig('find_outlier/GrLivArea_after.png')

var = "TotalBsmtSF"
# we can test this for outliers the same way, but we will skip it (no outliers)

# export df_train to csv
df_train.to_csv('train_clean.csv', index=False)

# one-hot encoding categorical variables
df_train = pd.get_dummies(df_train)