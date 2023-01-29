#import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, LassoCV, RidgeCV, ElasticNetCV)
from sklearn.metrics import mean_squared_error

# Referenced https://towardsdatascience.com/build-better-regression-models-with-lasso-271ce0f22bd

#Gather data and formate dataframes
def get_tables_for_year(year):
    #Gather data
    df = pd.read_csv("Investment_Data_Train (1).csv")
    df2 = pd.read_csv("Investment_Data_2020_withResponseVariable.xlsx - 2020_data.csv")
    new_df = pd.concat([df,df2], axis = "rows")[['MSN', 'StateCode', 'Year', 'Amount',
                 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments',
                 'TotalAmountofAssistance']]

    states_df = new_df[~new_df['StateCode'].isin(['DC', 'US', 'X3', 'X5'])]
    states_df2 = states_df[states_df['Year'] == year]

    #Formate dataframes
    MSN_df = states_df2.pivot(index="StateCode", columns=("MSN"), values="Amount")
    MSN_df.drop(['WDEXB', 'BDPRP', 'BFPRP', 'CLPRK', 'CLPRP', 'COPRK', 'ENPRP', 'NGMPK', 'NGMPP', 'PAPRP'], axis=1, inplace=True)
    Metrics_df = states_df2[['StateCode', 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments', 'TotalAmountofAssistance']]
    Metrics_df = Metrics_df.drop_duplicates(subset=None, keep="first", inplace=False)
    Metrics_df.set_index('StateCode', inplace=True)

    return MSN_df, Metrics_df.TotalAmountofAssistance

#See correlation between data
def plot_correlation(df):
    plt.figure(figsize=(16,10))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap="vlag", annot_kws={"size":6})
    plt.title("Correlation", fontsize=20)
    plt.show()

#Generate dataframe for all years
MSN_frames = []
metrics_frames = []
for year in range(2015, 2020):
    MSN_frames.append(get_tables_for_year(year)[0])
    metrics_frames.append(get_tables_for_year(year)[1])
all_MSN = pd.concat(MSN_frames)
all_metrics = pd.concat(metrics_frames)

#Set up dataframes for individual years
msn2015, metrics2015 = get_tables_for_year(2015)
msn2016, metrics2016 = get_tables_for_year(2016)
msn2017, metrics2017 = get_tables_for_year(2017)
msn2018, metrics2018 = get_tables_for_year(2018)
msn2019, metrics2019 = get_tables_for_year(2019)
msn2020, metrics2020 = get_tables_for_year(2020)

# Create and fit 2015 models
linear_regression15 = make_pipeline(StandardScaler(), LinearRegression())
linear_regression15.fit(msn2015, metrics2015)
lasso_model15 = make_pipeline(StandardScaler(), LassoCV())
lasso_model15.fit(msn2015, metrics2015)
ridge_model15 = make_pipeline(StandardScaler(), RidgeCV())
ridge_model15.fit(msn2015, metrics2015)

# Create and fit 2016 models
linear_regression16 = make_pipeline(StandardScaler(), LinearRegression())
linear_regression16.fit(msn2016, metrics2016)
lasso_model16 = make_pipeline(StandardScaler(), LassoCV())
lasso_model16.fit(msn2016, metrics2016)
ridge_model16 = make_pipeline(StandardScaler(), RidgeCV())
ridge_model16.fit(msn2016, metrics2016)

# Create and fit 2017 models
linear_regression17 = make_pipeline(StandardScaler(), LinearRegression())
linear_regression17.fit(msn2017, metrics2017)
lasso_model17 = make_pipeline(StandardScaler(), LassoCV())
lasso_model17.fit(msn2017, metrics2017)
ridge_model17 = make_pipeline(StandardScaler(), RidgeCV())
ridge_model17.fit(msn2017, metrics2017)

# Create and fit 2018 models
linear_regression18 = make_pipeline(StandardScaler(), LinearRegression())
linear_regression18.fit(msn2018, metrics2018)
lasso_model18 = make_pipeline(StandardScaler(), LassoCV())
lasso_model18.fit(msn2018, metrics2018)
ridge_model18 = make_pipeline(StandardScaler(), RidgeCV())
ridge_model18.fit(msn2018, metrics2018)

# Create and fit 2019 models
linear_regression19 = make_pipeline(StandardScaler(), LinearRegression())
linear_regression19.fit(msn2019, metrics2019)
lasso_model19 = make_pipeline(StandardScaler(), LassoCV())
lasso_model19.fit(msn2019, metrics2019)
ridge_model19 = make_pipeline(StandardScaler(), RidgeCV())
ridge_model19.fit(msn2019, metrics2019)


# Create and fit multi-year models
linear_regression_all = make_pipeline(StandardScaler(), LinearRegression())
linear_regression_all.fit(all_MSN, all_metrics)
lasso_all_model = make_pipeline(StandardScaler(), LassoCV())
lasso_all_model.fit(all_MSN, all_metrics)
ridge_all_model = make_pipeline(StandardScaler(), RidgeCV())
ridge_all_model.fit(all_MSN, all_metrics)


# Get MSE (Individual Years)
#2015
mse15 = mean_squared_error(metrics2020, abs(linear_regression15.predict(msn2020)),squared=False)
msel15 = mean_squared_error(metrics2020, abs(lasso_model15.predict(msn2020)),squared=False)
mser15 = mean_squared_error(metrics2020, abs(ridge_model15.predict(msn2020)),squared=False)

#2016
mse16 = mean_squared_error(metrics2020, abs(linear_regression16.predict(msn2020)),squared=False)
msel16 = mean_squared_error(metrics2020, abs(lasso_model16.predict(msn2020)),squared=False)
mser16 = mean_squared_error(metrics2020, abs(ridge_model16.predict(msn2020)),squared=False)

#2017
mse17 = mean_squared_error(metrics2020, abs(linear_regression17.predict(msn2020)),squared=False)
msel17 = mean_squared_error(metrics2020, abs(lasso_model17.predict(msn2020)),squared=False)
mser17 = mean_squared_error(metrics2020, abs(ridge_model17.predict(msn2020)),squared=False)

#2018
mse18 = mean_squared_error(metrics2020, abs(linear_regression18.predict(msn2020)),squared=False)
msel18 = mean_squared_error(metrics2020, abs(lasso_model18.predict(msn2020)),squared=False)
mser18 = mean_squared_error(metrics2020, abs(ridge_model18.predict(msn2020)),squared=False)

#2019
mse19 = mean_squared_error(metrics2020, abs(linear_regression19.predict(msn2020)),squared=False)
msel19 = mean_squared_error(metrics2020, abs(lasso_model19.predict(msn2020)),squared=False)
mser19 = mean_squared_error(metrics2020, abs(ridge_model19.predict(msn2020)),squared=False)

# Get MSE (All Years)
msea = mean_squared_error(metrics2020, abs(linear_regression_all.predict(msn2020)),squared=False)
msela = mean_squared_error(metrics2020, abs(lasso_all_model.predict(msn2020)),squared=False)
msera = mean_squared_error(metrics2020, abs(ridge_all_model.predict(msn2020)),squared=False)

# Print out results
print("MSEs")
print("Linear Regression 2015: " + "{:e}".format(mse15))
print("Linear Regression 2016: " + "{:e}".format(mse16))
print("Linear Regression 2017: " + "{:e}".format(mse17))
print("Linear Regression 2018: " + "{:e}".format(mse18))
print("Linear Regression 2019: " + "{:e}".format(mse19))
print("Linear Regression All: " + "{:e}".format(msea))

print("LASSO 2015: " + "{:e}".format(msel15))
print("LASSO 2016: " + "{:e}".format(msel16))
print("LASSO 2017: " + "{:e}".format(msel17))
print("LASSO 2018: " + "{:e}".format(msel18))
print("LASSO 2019: " + "{:e}".format(msel19))
print("LASSO All: " + "{:e}".format(msela))

print("Ridge 2015: " + "{:e}".format(mser15))
print("Ridge 2016: " + "{:e}".format(mser16))
print("Ridge 2017: " + "{:e}".format(mser17))
print("Ridge 2018: " + "{:e}".format(mser18))
print("Ridge 2019: " + "{:e}".format(mser19))
print("Ridge_All MSE: " + "{:e}".format(msera))


# Graph some weights
# Weights
lasso_model_coef = lasso_all_model[-1].coef_
print("Weights for LASSO Model (Multiyear)")
print(list(zip(all_MSN.columns, lasso_model_coef)))

ridge_all_model_coef = ridge_all_model[-1].coef_
print("Weights for Ridge Model Model (Multiyear)")
print(list(zip(all_MSN.columns, ridge_all_model_coef)))

linear_regression_coef = linear_regression_all[-1].coef_
print("Weights for Linear Regression Model (Multiyear)")
print(list(zip(all_MSN.columns, linear_regression_coef)))

def weights_graph(model):
    model_coef = model[-1].coef_
    plt.figure(figsize=(16,10))
    plt.bar(msn2015.columns, model_coef, color='maroon', width=0.4)
    plt.xticks(fontsize=5, rotation=-45)
    plt.title("Weight of Each Factor")
    plt.ylabel("Weights")
    plt.xlabel("Factors")
    plt.yscale("linear")
    plt.show()

plt.figure(figsize=(16,10))
plt.bar(["Regression 15","LASSO 15","Ridge 15",
         "Regression 16","LASSO 16","Ridge 16",
         "Regression 17","LASSO 17","Ridge 17",
         "Regression 18","LASSO 18","Ridge 18",
         "Regression 19","LASSO 19","Ridge 19",
         "Regression All","LASSO All","Ridge All"], [mse15,msel15,mser15,
                                                     mse16,msel16,mser16,
                                                     mse17,msel17,mser17,
                                                     mse18,msel18,mser18,
                                                     mse19,msel19,mser19,
                                                     msea,msela,msera], color='maroon', width=0.4)
plt.ylim(2.5*10**7,6*10**7)
plt.xticks(fontsize=5, rotation = -45)
plt.title("RMSE for Each Model")
plt.ylabel("RMSE")
plt.yscale("log")
plt.show()


weights_graph(lasso_all_model)
weights_graph(ridge_all_model)
weights_graph(linear_regression_all)

print("-----PREDICTIONS-----")
print(abs(lasso_all_model.predict(msn2020)))