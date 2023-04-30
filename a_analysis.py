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
print(df_train.columns)

# # analyze target
# print(df_train['SalePrice'].describe())
# plot = sns.distplot(df_train['SalePrice'])
# fig = plot.get_figure()
# fig.savefig('analysis_figs/SalePrice.png')

# # analyze variables one by one (a little subjective, dont like it)
# var = "GrLivArea" 
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.savefig('analysis_figs/GrLivArea.png')

# var2 = "TotalBsmtSF"
# data2 = pd.concat([df_train['SalePrice'], df_train[var2]], axis=1)
# data2.plot.scatter(x=var2, y='SalePrice', ylim=(0, 800000))
# plt.savefig('analysis_figs/TotalBsmtSF.png')

# var3 = "OverallQual"
# data3 = pd.concat([df_train['SalePrice'], df_train[var3]], axis=1)
# fig = sns.boxplot(x=var3, y="SalePrice", data=data3)
# fig.axis(ymin=0, ymax=800000)
# plt.savefig('analysis_figs/OverallQual.png')

# var4 = "YearBuilt"
# data4 = pd.concat([df_train['SalePrice'], df_train[var4]], axis=1)
# fig = sns.boxplot(x=var4, y="SalePrice", data=data4)
# fig.axis(ymin=0, ymax=800000)
# plt.savefig('analysis_figs/YearBuilt.png')

# correalation analysis: 
#   correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9)) # figsize: width, height in inches
fig = sns.heatmap(corrmat, vmax=.8, square=True)
plt.savefig('analysis_figs/corrmat.png')

# get the top 10 in correlation matrix and create another heatmap
k = 10  
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
corrmat2  = df_train[cols].corr()
fig = sns.heatmap(corrmat2, annot=True, square=True)
plt.savefig('analysis_figs/corrmat2.png')

# scatterplot all the candidates
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.savefig('analysis_figs/scatterplot.png')