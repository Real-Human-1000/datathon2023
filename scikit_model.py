import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from time import sleep

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, MultiTaskLassoCV, RidgeCV)
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
    Metrics_df = states_df2[['StateCode', 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments', 'TotalAmountofAssistance']]
    Metrics_df = Metrics_df.drop_duplicates(subset=None, keep="first", inplace=False)
    Metrics_df.set_index('StateCode', inplace=True)
    # Final_df = pd.concat([MSN_df, Metrics_df], axis="columns")

    return MSN_df, Metrics_df


def plot_correlation(df):
    plt.figure(figsize=(16,10))

    sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap="vlag", annot_kws={"size":6})

    plt.title("Correlation", fontsize=20)

    plt.show()


msn2015, metrics2015 = get_tables_for_year(2015)
msn2016, metrics2016 = get_tables_for_year(2016)

# print(msn2015)
# print(msn2016)

# plot_correlation(msn2015)

linear_regression = make_pipeline(StandardScaler(), LinearRegression())
linear_regression.fit(msn2015, metrics2015)

mse = mean_squared_error(metrics2016, linear_regression.predict(msn2016))
print(mse)

# linear_regression_coef = linear_regression[-1].coef_

# print(list(linear_regression_coef))


lasso_cv = make_pipeline(StandardScaler(), MultiTaskLassoCV())
lasso_cv.fit(msn2015, metrics2015)

mse = mean_squared_error(metrics2016, lasso_cv.predict(msn2016))
print(mse)

