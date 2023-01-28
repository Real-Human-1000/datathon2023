class LinearModel:
    """
    A class used to represent a Linear statistical
    model of multiple variables. This model takes
    a vector of input variables and predicts that
    the measured variable will be their weighted sum.
    """

    def __init__(self, weights):
        """
        Create a new LinearModel.

        inputs:
            - weights: an m x 1 matrix of weights
        """
        self._weights = weights

    def __str__(self):
        """
        Return: weights as a human readable string.
        """
        return str(self._weights)

    def get_weights(self):
        """
        Return: the weights associated with the model.
        """
        return self._weights

    def generate_predictions(self, inputs):
        """
        Use this model to predict a matrix of
        measured variables given a matrix of input data.

        inputs:
            - inputs: an n x m matrix of explanatory variables

        Returns: an n x 1 matrix of predictions
        """
        weights = self.get_weights()
        matrix = inputs @ weights
        return matrix

    def prediction_error(self, inputs, actual_result):
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
        prediction_matrix = self.generate_predictions(inputs)
        vals = prediction_matrix.shape()
        rows = vals[0]

        # initialize the mse
        mse_total = 0

        # add the squared error for each data to the mse
        for curr_row in range(rows):
            pred_val = prediction_matrix[(curr_row, 0)]
            actual_val = actual_result[(curr_row, 0)]
            error = actual_val - pred_val
            squared_error = error ** 2
            mse_total += squared_error

        # compute the mse
        mse = mse_total / rows

        return mse