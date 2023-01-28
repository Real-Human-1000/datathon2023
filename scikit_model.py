import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from time import sleep

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, Lasso, RidgeCV)
from sklearn.metrics import mean_squared_error

# Referenced https://towardsdatascience.com/build-better-regression-models-with-lasso-271ce0f22bd


def get_tables_for_year(year):
    df = pd.read_csv("Investment_Data_Train (1).csv")
    new_df = df[['MSN', 'StateCode', 'Year', 'Amount',
                 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments',
                 'TotalAmountofAssistance']]

    states_df = new_df[~new_df['StateCode'].isin(['DC', 'US', 'X3', 'X5'])]
    states_df2 = states_df[states_df['Year'] == year]

    MSN_df = states_df2.pivot(index="StateCode", columns=("MSN"), values="Amount")
    MSN_df.drop(['WDEXB', 'BDPRP', 'BFPRP', 'CLPRP', 'COPRK', 'ENPRP', 'NGMPK', 'NGMPP', 'PAPRP'], axis=1, inplace=True)
    Metrics_df = states_df2[['StateCode', 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments', 'TotalAmountofAssistance']]
    Metrics_df = Metrics_df.drop_duplicates(subset=None, keep="first", inplace=False)
    Metrics_df.set_index('StateCode', inplace=True)

    return MSN_df, Metrics_df.TotalAmountofAssistance

"""
new_df = df[['MSN', 'StateCode', 'Year', 'Amount',
             'CO2 Emissions (Mmt)', 'TotalNumberofInvestments',
             'TotalAmountofAssistance']]

states_df2 = new_df[~new_df['StateCode'].isin(['DC', 'US', 'X3', 'X5'])]

MSN_df = states_df2.pivot(index="StateCode", columns=("MSN"), values="Amount")
MSN_df.drop(['WDEXB', 'BDPRP', 'BFPRP', 'CLPRP', 'COPRK', 'ENPRP', 'NGMPK', 'NGMPP', 'PAPRP'], axis=1, inplace=True)
Metrics_df = states_df2[['StateCode', 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments', 'TotalAmountofAssistance']]
Metrics_df = Metrics_df.drop_duplicates(subset=None, keep="first", inplace=False)
Metrics_df.set_index('StateCode', inplace=True)
"""

def reorganize_msn_metrics(msn, metrics):
    data = msn.assign(CO2Emissions=metrics[['CO2 Emissions (Mmt)']])
    data = data.assign(TotalInvestments=metrics[['TotalNumberofInvestments']])
    output = metrics[['TotalAmountofAssistance']]
    return data, output


def plot_correlation(df):
    plt.figure(figsize=(16,10))

    sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap="vlag", annot_kws={"size":6})

    plt.title("Correlation", fontsize=20)

    plt.show()


msn2015, metrics2015 = get_tables_for_year(2015)
data2015, out2015 = get_tables_for_year(2015)

msn2016, metrics2016 = get_tables_for_year(2016)
data2016, out2016 = get_tables_for_year(2016)

# plot_correlation(data2015)

linear_regression = make_pipeline(StandardScaler(), LinearRegression())
linear_regression.fit(msn2015, out2015)

mse = mean_squared_error(out2016, linear_regression.predict(msn2016))
print("{:e}".format(mse))

# linear_regression_coef = linear_regression[-1].coef_
# print(list(linear_regression_coef))


lasso_model = make_pipeline(StandardScaler(), Lasso(tol=0.01, max_iter=100000))
lasso_model.fit(msn2015, out2015)
ridge_model = make_pipeline(StandardScaler(), RidgeCV())
ridge_model.fit(msn2015, out2015)
# lasso_model_coef = lasso_model[-1].coef_
# print(list(lasso_model_coef))

print(ridge_model.predict(msn2016))
mse = mean_squared_error(out2016, lasso_model.predict(msn2016))
mse2 = mean_squared_error(out2016, ridge_model.predict(msn2016))
print("{:e}".format(mse))
print("{:e}".format(mse2))
