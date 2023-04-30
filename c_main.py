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