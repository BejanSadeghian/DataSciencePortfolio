import numpy as np
import statsmodels.api as sm
###########
## Build ##
###########
class marketModel(object):
    """
    Purpose: Properly transform the input data for market modeling and allow for modeling
    Methods:
        1. Transform input data by weighting according to volume share
        2. Model using the GLM market model with fourier series seasonality components
        3. Predict based on fitted model
    """
    def __init__(self):
        """
        Returns an instance of marketModel object

        Attributes:
            1. Coefficients -> Extracted from GLM for all params {type: array}
            2. chisq -> Extracted chisq results from GLM {type: float}
            3. pvalues -> Calculated p_values of coefficients {type: float}
            4. Objective Function -> Objective function used for GLM {type: statsmodels object}
            5. Model -> SM Model Object {type: statsmodels object}
            6. constCreated -> Bool to track if a constant variable was used for modeling, if so then prediction will also use it {type: bool}
            7. predictRMSE -> If giving a target array to the predict method then this is the calculated RMSE between predicted and actual {type: float}
        """
        self.coef_ = None
        self.chisq = None
        self.pvalues = None
        self.objFunc = None
        self.model = None
        self.constCreated = None
        self.predictRMSE = None

    def transform(self, weightData, inputDataArr, by_row = True, add_fourier = False, A = None, trend = None, add_const = True):
        """
        Returns predictor data that is weighed according to weightData proportions and the aggregated target array
        Notes: weightData shape must match each inputDataArr object's shape either row-wise or col-wise depending on by_row parameter
            Fourier series components come in pairs sin(2*pi*a*trend)+cos(2*pi*a*trend) for each value of a

        Parameters:
            1. weightData -> Data to weight input variables according to proportion of total {type: matrix}
            2. inputDataArr -> Array of data matrices to transform and return {type: array of matrices}
            3. by_row -> bool to determine if weightData proportion and data transformation should be row sum or col sum based {type: bool}
            4. add_fourier -> bool to determine if we should add fourier series components to the predictor variables {type: bool}
            5. A -> list of range of a values for the fourier series if used {type: list of ints}
            6. trend -> array of size (n,1) where n is the same as the weightData and inputDatasets representing trend data {type: array}
            7. add_const -> Bool to add a constant variable to predictor matrix {type: bool}
        Returns:
            1. Array for target data (summation)
            2. Matrix of predictors (weighted average)
        """
        n,m = np.shape(weightData)
        self.constCreated = add_const
        #Size check
        if sum([np.shape(x) == (n,m) for x in inputDataArr]) != len(inputDataArr):
            raise ValueError

        #Scaling Matrix Calculation
        if by_row:
            scaling = weightData / np.multiply(np.sum(weightData, axis=1, keepdims=True), np.ones((1,m)))
        else:
            scaling = weightData / np.multiply(np.ones((n,m)), np.sum(weightData, axis=0, keepdims=True))

        #Transform inputDataArr
        output = list()
        for data in inputDataArr:
             output.append(np.nansum(np.multiply(data,scaling),axis=1))

        if add_fourier:
            for a in A:
                sin = np.sin(np.multiply(2 * np.pi * a, trend))
                output.append(sin)
            for a in A:
                cos = np.cos(np.multiply(2 * np.pi * a, trend))
                output.append(cos)

        if self.constCreated:
            output.append(np.repeat(1,n))

        return (np.nansum(weightData,axis=1), np.transpose(np.vstack(output)))

    def fitModel(self, target, predictors, objective = sm.robust.norms.TukeyBiweight(c=4.685)):
        """
        Fits the GLM to the data and saves the data to the object.

        Parameters:
            1. target -> Matrix of target data {type: matrix}
            2. predictors -> Array of predictor data to model {type: array of matrices}
            3. add_const -> Bool to add a constant variable if wanted {type: bool}

        Returns:
            None

        Saves to attributes:
            1. coef_ -> Coefficients of fitted GLM
            2. chisq - > Chisq results of GLM
            3. pvalues -> pvalues of the coefficients
            4. objFunc -> Objective Function of GLM fitted
            5. model -> model object from fitted GLM
        """
        n,m = np.shape(predictors)

        #Fit the model and save the outputs
        self.objFunc = objective
        self.model = sm.RLM(target, predictors, M=self.objFunc).fit()
        self.coef_ = self.model.params
        self.pvalues = self.model.pvalues
        self.chisq = self.model.chisq
        return None

    def predict(self, predictors, target = None):
        """
        Predicts the outcome and calculates RMSE and residuals based on fitted model.

        Parameters:
            1. predictors -> Array of predictor data to predict from {type: array of matrices}
            2. target -> Matrix of target data to check against {type: matrix}

        Returns:
            1. Array of predictions based on input predictors data

        Saves to attributes:
            1. predictRMSE -> Saves calculated RMSE if a target array is provided
        """
        if self.model is None: #Check if model was fitted
            raise Exception("Model needs to be fitted first.")
        else:
            prediction = np.dot(predictors,self.coef_)
        if not(target is None):
            self.predictRMSE = np.sqrt(np.mean((target - prediction)**2))
        return prediction
