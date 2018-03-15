"""
@title: backwards selection of OLS using P-Value as the selection criteria

@author: Bejan Lee Sadeghian
@date: March 8th, 2018
"""

import statsmodels.api as sm
import numpy as np

class OLS_BackwardsSelection(object):
    def __init__(self, exog, endog):
        """A custom built backwards selection object based on p-value"""
        self.modelX = exog.values
        self.modelY = endog
        self.variables = list(exog.columns.values)
        self.modelXBack = None
        
        self.model = None
        self.results = None
        self.coef = None
        self.Rsquared = None
        self.Rsquared_adj = None
        return None
    
    def modelOLS(self, y=None, x=None):
        """Creates a OLS model instance"""
        if x == None or y == None:
            x = self.modelX
            y = self.modelY
        self.model = sm.OLS(y,x)
        self.results = self.model.fit()
        self.coef = self.results.params
        self.Rsquared = self.results.rsquared
        self.Rsquared_adj = self.results.rsquared_adj
        return None
    
    def _removeColumn(self, colNum):
        """Removes variables based on the index the dataset that is being edited"""
        del self.variables[colNum]
        self.modelXBack = np.delete(self.modelXBack, colNum, axis=1)
        return None
    
    def backwardSelect(self, criteria = 0.05, maxSteps = 10):
        """This backwards selection mechanism uses P-Value to remove the least significant variable"""
        self.modelXBack = self.modelX.copy() #initialize modelXBack
        iteration = 0
        while sum(self.results.pvalues > criteria) != 0 and maxSteps >= iteration and self.Rsquared > 0.7:
            indexRemove = np.where(self.results.pvalues == max(self.results.pvalues))[0][0] #Get the first index
            self._removeColumn(colNum=indexRemove)
            #Remodel with only the selected features
            self.modelOLS(y=self.modelY, x=self.modelXBack)
            iteration = iteration + 1
            
    def predict(self, exogTest):
        """Predicts with the current instance of the model"""
        return self.results.predict(exogTest.loc[:,self.variables])