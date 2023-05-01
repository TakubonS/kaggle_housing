import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

def show_missing_status(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def prepare_X(df):
    df["GrLivArea"] = np.log(df["GrLivArea"])
    df['TotalBsmtSF'] = np.where(df['TotalBsmtSF'] > 0, np.log(df['TotalBsmtSF']), df['TotalBsmtSF'])
    df = pd.get_dummies(df)
    ret = df[["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]]
    return ret


df_train = pd.read_csv('train_clean_normalized.csv')
corrmat = df_train.corr()
k = 10  
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
corrmat2  = df_train[cols].corr()
fig = sns.heatmap(corrmat2, annot=True, square=True)
plt.savefig('new_analysis_fig/corrmat2.png')

X = df_train[["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]]
y = df_train['SalePrice']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(train_X, train_y)
y_pred = lr.predict(test_X)
loss = mean_squared_error(test_y, y_pred)
print("loss: ", loss)

test_df = pd.read_csv('test.csv')
submission_X = prepare_X(test_df)
submission_X["GarageCars"].fillna(submission_X["GarageCars"].mean(), inplace=True)
submission_X["TotalBsmtSF"].fillna(submission_X["TotalBsmtSF"].mean(), inplace=True)
submission_y = lr.predict(submission_X)
submission_y = np.exp(submission_y)
submission_df = pd.DataFrame({
    "ID": test_df["Id"].values,
    "SalePrice": submission_y
})
submission_df.to_csv("./submission.csv", index = False)
