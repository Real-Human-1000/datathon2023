import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, LassoCV, RidgeCV, ElasticNetCV)
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
    MSN_df.drop(['WDEXB', 'BDPRP', 'BFPRP', 'CLPRK', 'CLPRP', 'COPRK', 'ENPRP', 'NGMPK', 'NGMPP', 'PAPRP'], axis=1, inplace=True)
    Metrics_df = states_df2[['StateCode', 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments', 'TotalAmountofAssistance']]
    Metrics_df = Metrics_df.drop_duplicates(subset=None, keep="first", inplace=False)
    Metrics_df.set_index('StateCode', inplace=True)

    # MSN_df = MSN_df.assign(CO2Emissions=Metrics_df[['CO2 Emissions (Mmt)']])
    # MSN_df = MSN_df.assign(CO2Emissions=Metrics_df[['TotalNumberofInvestments']])

    return MSN_df, Metrics_df.TotalAmountofAssistance


def plot_correlation(df):
    plt.figure(figsize=(16,10))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap="vlag", annot_kws={"size":6})
    plt.title("Correlation", fontsize=20)
    plt.show()


MSN_frames = []
metrics_frames = []
for year in range(2015, 2020):
    MSN_frames.append(get_tables_for_year(year)[0])
    metrics_frames.append(get_tables_for_year(year)[1])
#creates MSN matrix for years from 2015-2018
MSN_matrix, Metrics_matrix = get_tables_for_year(2015)
MSN_matrix16, Metrics_matrix16 = get_tables_for_year(2016)
all_MSN = pd.concat(MSN_frames)
all_metrics = pd.concat(metrics_frames)

msn2015, metrics2015 = get_tables_for_year(2015)
data2015, out2015 = get_tables_for_year(2015)
msn2016, metrics2016 = get_tables_for_year(2019)
data2016, out2016 = get_tables_for_year(2016)

msn2019, out2019 = get_tables_for_year(2015)
# plot_correlation(data2015)

# Create and fit standard (single-year) models
linear_regression = make_pipeline(StandardScaler(), LinearRegression())
linear_regression.fit(msn2015, out2015)
lasso_model = make_pipeline(StandardScaler(), LassoCV())
lasso_model.fit(msn2015, out2015)
ridge_model = make_pipeline(StandardScaler(), RidgeCV())
ridge_model.fit(msn2015, out2015)
elasticnet_model = make_pipeline(StandardScaler(), ElasticNetCV())
elasticnet_model.fit(msn2015, out2015)

# Create and fit multi-year models
linear_regression_all = make_pipeline(StandardScaler(), LinearRegression())
linear_regression_all.fit(all_MSN, all_metrics)
lasso_all_model = make_pipeline(StandardScaler(), LassoCV())
lasso_all_model.fit(all_MSN, all_metrics)
ridge_all_model = make_pipeline(StandardScaler(), RidgeCV())
ridge_all_model.fit(all_MSN, all_metrics)
elasticnet_all_model = make_pipeline(StandardScaler(), ElasticNetCV())
elasticnet_all_model.fit(all_MSN, all_metrics)

# Get MSE
mse = mean_squared_error(out2016, linear_regression.predict(msn2016))
msea = mean_squared_error(out2019, linear_regression_all.predict(msn2019))
msel = mean_squared_error(out2016, lasso_model.predict(msn2016))
msela = mean_squared_error(out2019, lasso_all_model.predict(msn2019))
mser = mean_squared_error(out2016, ridge_model.predict(msn2016))
msera = mean_squared_error(out2019, ridge_all_model.predict(msn2019))
msee = mean_squared_error(out2016, elasticnet_model.predict(msn2019))
mseea = mean_squared_error(out2019, elasticnet_all_model.predict(msn2019))

# Print out results
print("Linear Regression MSE: " + "{:e}".format(mse))
print("Linear Regression_All MSE: " + "{:e}".format(msea))
print("LASSO MSE: " + "{:e}".format(msel))
print("LASSO_All MSE: " + "{:e}".format(msela))
print("Ridge MSE: " + "{:e}".format(mser))
print("Ridge_All MSE: " + "{:e}".format(msera))
print("ElasticNet MSE: " + "{:e}".format(msee))
print("ElasticNet_All MSE: " + "{:e}".format(mseea))


# Graph some weights
# Weights
lasso_model_coef = lasso_all_model[-1].coef_
print(list(zip(msn2015.columns, lasso_model_coef)))

elasticnet_all_model_coef = elasticnet_all_model[-1].coef_
print(list(zip(msn2015.columns, elasticnet_all_model_coef)))

ridge_all_model_coef = ridge_all_model[-1].coef_
print(list(zip(msn2015.columns, ridge_all_model_coef)))


def weights_graph(model):
    model_coef = list(model[-1].coef_)
    print(model_coef)
    plt.figure(figsize=(16,10))
    plt.bar(msn2015.columns, model_coef, color='maroon', width=0.4)
    plt.show()


weights_graph(lasso_model)