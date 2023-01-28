import pandas as pd
import numpy as np
import numpy.matrixlib as mat
from numpy.linalg import inv
from LinearModel import LinearModel


def get_tables_for_year(year):
    df = pd.read_csv("Investment_Data_Train (1).csv")
    new_df = df[['MSN', 'StateCode', 'Year', 'Amount',
                 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments',
                 'TotalAmountofAssistance']]

    states_df = new_df[~new_df['StateCode'].isin(['DC', 'US', 'X3', 'X5'])]
    states_df2 = states_df[states_df['Year'] == year]

    MSN_df = states_df2.pivot(index="StateCode", columns=("MSN"), values="Amount")
    MSN_df.drop('WDEXB', axis=1, inplace=True)
    Metrics_df = states_df2[['StateCode', 'CO2 Emissions (Mmt)', 'TotalNumberofInvestments', 'TotalAmountofAssistance']]
    Metrics_df = Metrics_df.drop_duplicates(subset=None, keep="first", inplace=False)
    Metrics_df.set_index('StateCode', inplace=True)
    # Final_df = pd.concat([MSN_df, Metrics_df], axis="columns")

    return MSN_df.to_numpy(), Metrics_df.TotalAmountofAssistance.to_numpy()


MSN_matrix, Metrics_matrix = get_tables_for_year(2015)
MSN_matrix16, Metrics_matrix16 = get_tables_for_year(2016)

print(np.shape(Metrics_matrix16))


def fit_least_squares(input_data, output_data):
    """
    Create a Linear Model which predicts the output vector
    given the input matrix with minimal Mean-Squared Error.

    inputs:
        - input_data: an n x m matrix
        - output_data: an n x 1 matrix

    returns: a LinearModel object which has been fit to approximately
    match the data
    """

    # solve for the weights
    input_transpose = np.transpose(input_data)
    input_mul = input_transpose @ input_data
    deter = np.linalg.det(input_mul)
    input_inverse = np.linalg.inv(input_mul)
    weights = (input_inverse @ input_transpose) @ output_data

    return LinearModel(weights)


def soft_threshold(x_val, t_val):
    '''
    x_val: a float
    t_val: a float

    outputs: a float. This float is the result of moving
    x closer to 0 by the value t
    '''

    # move x closer to 0 by value t
    if x_val > t_val:
        return x_val - t_val
    elif abs(x_val) <= t_val:
        return 0
    else:
        return x_val + t_val


def fit_lasso(param, iterations, input_data, output_data):
    """
    Create a Linear Model which predicts the output vector
    given the input matrix using the LASSO method.

    inputs:
        - param: a float representing the lambda parameter
        - iterations: an integer representing the number of iterations
        - input_data: an n x m matrix
        - output_data: an n x 1 matrix

    returns: a LinearModel object which has been fit to approximately
    match the data
    """


    # set up the matrixes necessary for the computation
    vals = np.shape(input_data)
    cols = vals[1]

    lse_fit = fit_least_squares(input_data, output_data)
    lse = lse_fit.get_weights()
    print(np.shape(input_data))

    matrix1 = np.transpose(input_data) @ output_data
    matrix2 = np.transpose(input_data) @ input_data

    curr_iter = 0

    # computations necessary for lasso algorithm
    while curr_iter < iterations:

        lse_old = np.copy(lse)

        for curr_col in range(cols):
            matrix3 = matrix2[curr_col,:] @ lse
            a_j = (matrix1[curr_col] - matrix3) / matrix2[curr_col, curr_col]
            b_j = param / (2.0 * matrix2[curr_col, curr_col])
            lse[curr_col] = soft_threshold(lse[curr_col] + a_j, b_j)

        diff_matrix = lse - lse_old
        diff_abs = np.absolute(diff_matrix)
        diff_sum = np.sum(diff_abs)

        # if within a certain threshold stop algorithm
        if diff_sum < 0.00001:
            break

        curr_iter += 1

    return LinearModel(lse)


def run_experiment(iterations):
    """
    Using some historical data from 1954-2000, as
    training data, generate weights for a Linear Model
    using both the Least-Squares method and the
    LASSO method (with several different lambda values).

    Test each of these models using the historical
    data from 2001-2012 as test data.

    inputs:
        - iterations: an integer representing the number of iterations to use

    Print out the model's prediction error on the two data sets
    """

    # set up the models
    lsemodel = fit_least_squares(MSN_matrix, Metrics_matrix)
    lassomodel = fit_lasso(1000, iterations, MSN_matrix, Metrics_matrix)
    lassomodel2 = fit_lasso(50000, iterations, MSN_matrix, Metrics_matrix)
    lassomodel3 = fit_lasso(10000000, iterations, MSN_matrix, Metrics_matrix)

    # print the prediction error for each model based on the 2001-2012 data
    #print("lse | training: ", lsemodel.prediction_error(stats5400, wins5400))
    #print("LASSO model | λ of 1000 | training: ", lassomodel.prediction_error(stats5400, wins5400))
    #print("LASSO model | λ of 50000 | training: ", lassomodel2.prediction_error(stats5400, wins5400))
    #print("LASSO model | λ of 100000 | training: ", lassomodel3.prediction_error(stats5400, wins5400))

    # print the prediction error for each model based on the 2001-2012 data
    print("lse | testing: ", lsemodel.prediction_error(MSN_matrix16, Metrics_matrix16))
    print("LASSO model | λ of 1000 | testing: ", lassomodel.prediction_error(MSN_matrix16, Metrics_matrix16))
    print("LASSO model | λ of 50000 | testing: ", lassomodel2.prediction_error(MSN_matrix16, Metrics_matrix16))
    print("LASSO model | λ of 100000 | testing: ", lassomodel3.prediction_error(MSN_matrix16, Metrics_matrix16))

    print("LASSO model | λ of 1000 | testing: ", lassomodel.prediction_error(MSN_matrix, Metrics_matrix))
    print(lassomodel.generate_predictions(MSN_matrix))

run_experiment(10000)
