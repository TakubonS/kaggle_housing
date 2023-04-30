import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#  https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python

df_train = pd.read_csv('train_clean.csv')

# Normality: make sure variables are normally distributed
#   SalesPrice  
#       histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
plt.savefig('normality/SalePrice_before1.png')
fig = plt.figure() # new canvas
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.savefig('normality/SalePrice_before2.png')

# not normal, so apply log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
fig = plt.figure() # new canvas
sns.distplot(df_train['SalePrice'], fit=norm)
plt.savefig('normality/SalePrice_after1.png')
fig = plt.figure() # new canvas
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.savefig('normality/SalePrice_after2.png')

#    GrLivArea
fig = plt.figure() # new canvas
sns.distplot(df_train['GrLivArea'], fit=norm)
plt.savefig('normality/GrLivArea_before1.png')
fig = plt.figure() # new canvas
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.savefig('normality/GrLivArea_before2.png')

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
fig = plt.figure() # new canvas
sns.distplot(df_train['GrLivArea'], fit=norm)
plt.savefig('normality/GrLivArea_after1.png')
fig = plt.figure() # new canvas
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.savefig('normality/GrLivArea_after2.png')

#   TotalBsmtSF
fig = plt.figure() # new canvas
sns.distplot(df_train['TotalBsmtSF'], fit=norm)
plt.savefig('normality/TotalBsmtSF_before1.png')
fig = plt.figure() # new canvas
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
plt.savefig('normality/TotalBsmtSF_before2.png')

#   Do log transformation for all non-zero data
df_train['TotalBsmtSF'] = np.where(df_train['TotalBsmtSF'] > 0, np.log(df_train['TotalBsmtSF']), df_train['TotalBsmtSF'])
fig = plt.figure() # new canvas
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
plt.savefig('normality/TotalBsmtSF_after1.png')
fig = plt.figure() # new canvas
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.savefig('normality/TotalBsmtSF_after2.png')