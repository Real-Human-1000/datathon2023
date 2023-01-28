import pandas as pd
nba = pd.read_csv("Investment_Data_Train (1).csv")
print(nba)

print("test")


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
    input_transpose = input_data.transpose()
    input_mul = input_transpose @ input_data
    input_inverse = input_mul.inverse()
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
    vals = input_data.shape()
    cols = vals[1]

    lse_fit = fit_least_squares(input_data, output_data)
    lse = lse_fit.get_weights()

    matrix1 = input_data.transpose() @ output_data
    matrix2 = input_data.transpose() @ input_data

    curr_iter = 0

    # computations necessary for lasso algorithm
    while curr_iter < iterations:

        lse_old = lse.copy()

        for curr_col in range(cols):
            matrix3 = matrix2.getrow(curr_col) @ lse
            a_j = (matrix1[(curr_col, 0)] - matrix3[(0, 0)]) / matrix2[(curr_col, curr_col)]
            b_j = param / (2.0 * matrix2[(curr_col, curr_col)])
            lse[(curr_col, 0)] = soft_threshold(lse[(curr_col, 0)] + a_j, b_j)

        diff_matrix = lse - lse_old
        diff_abs = diff_matrix.abs()
        diff_sum = diff_abs.summation()

        # if within a certain threshold stop algorithm
        if diff_sum < 0.00001:
            break

        curr_iter += 1

    return LinearModel(lse)
