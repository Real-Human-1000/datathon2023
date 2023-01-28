import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor


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
MSN_frames = []
metrics_frames = []
for year in range(2015, 2019):
    MSN_frames.append(get_tables_for_year(year)[0])
    metrics_frames.append(get_tables_for_year(year)[1])
#creates MSN matrix for years from 2015-2018
MSN_matrix, Metrics_matrix = get_tables_for_year(2015)
MSN_matrix16, Metrics_matrix16 = get_tables_for_year(2016)
all_MSN = pd.concat(MSN_frames)
all_metrics = pd.concat(metrics_frames)


msn2015, metrics2015 = get_tables_for_year(2015)
msn2016, metrics2016 = get_tables_for_year(2016)

np.random.seed(42)

X_train = msn2015
y_train = metrics2015

X_test = msn2016
y_test = metrics2016

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train,y_train)

print(rfr.score(X_test, y_test))

print(rfr.predict(X_test))


def prediction_error(predictions, actual_result):
    """
    Calculate the MSE between the actual measured
    data and the predictions generated by this model
    based on the input data.

    inputs:
        - inputs: inputs: an n x m matrix of explanatory variables
        - actual_result: an n x 1 matrix of the corresponding
                         actual values for the measured variables

    Returns: a float that is the MSE between the generated
    data and the actual data
    """

    # generate the predictions
    prediction_matrix = predictions
    vals = np.shape(prediction_matrix)
    rows = vals[0]

    # initialize the mse
    mse_total = 0

    # add the squared error for each data to the mse
    for curr_row in range(rows):
        pred_val = prediction_matrix[curr_row]
        actual_val = actual_result[curr_row]
        error = actual_val - pred_val
        squared_error = error ** 2
        mse_total += squared_error

    # compute the mse
    mse = mse_total / rows

    return mse


preds = rfr.predict(X_test)
vals = math.sqrt(prediction_error(preds,y_test))
print("{:e}".format(vals**2))
