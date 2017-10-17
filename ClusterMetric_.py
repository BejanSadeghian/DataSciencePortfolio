###############################
## Purpose: ClusterMetric Object Definition
## Author: Bejan Lee Sadeghian
## Date: 10/15/2017
###############################

import numpy as np

class ClusterMetric(object):
    """
    A Object to calculate various cluster metrics using the attributes of a KMeans Instance.
    
    Attributes:
        1. Cluter Centroids
        2. Cluster Memberships
        3. Cluster Observation Data
        4. Within Sum of Squares (WSS) Metric
        5. Between Sum of Squares (BSS) Metric
        6. Total Sum of Squares (TSS) Metric
        7. Cohesion Metric    #TODO: Future Addition
        8. Separation Metric    #TODO: Future Addition
        9. Silhouette Metric    #TODO: Future Addition
        
    External Methods:
        1. "fit" -> Fits current data to our instance and calculates WSS, BSS, TSS Attributes
    
    Internal Methods:
        1. "squareVector" -> Calculates the Sum of Squared Error for a observation vector
        2. "getWSS" -> Calculates the WSS metric for our fitted data
        3. "getTSS" -> Calculates the TSS metric for our fitted data
    """
    def __init__(self):
        """
        Returns an instance of ClusterMetric that contains attributes of our cluster results
        
        Parameters:
            1. cohesion = When True, calculate cohesion metric within each cluster
            2. separation = When True, calculate separation metric within each cluster
        """
        # Initialize dataset attributes
        self.centroids = None
        self.memberships = None
        self.obs_data = None
        
        # Initialize Metrics
        self.WSS = None
        self.BSS = None
        self.TSS = None
      
    def _squareVector(self, vector):
        """Calculates the squared sum of the vector provided"""
        return np.matmul(vector, vector.T)
    
    def _getWSS(self):
        """
        Performing WSS Calculation by applying squareVector due to likely memory constraints.
        
        Returns:
            Within Sum of Squares (WSS) Metric
        """
        within_error = self.obs_data - self.centroids[self.memberships]
        self.WSS = sum(np.apply_along_axis(self._squareVector, axis=1, arr = within_error))
        
        return self.WSS
    
    def _getTSS(self):
        """
        Performing TSS Calculation for all centroids

        Returns:
            Total Sum of Squares (TSS) Metric
        """
        total_error = self.obs_data - np.tile(self.obs_data.mean(axis=0), (self.obs_data.shape[0],1))
        self.TSS = sum(np.apply_along_axis(self._squareVector, axis=1, arr = total_error))
        
        return self.TSS
    
    def fit(self, cl_centroids, cl_memberships, cl_data):
        """
        Calculates the WSS, BSS, TSS metrics using our observation data
  
        Parameters:
            cl_centroids = cluster centroid list or array
            cl_memberships = cluster membership list for all observations in the data
            cl_data = the data used to cluster on
        """
        self.centroids = np.array(cl_centroids)
        self.memberships = np.array(cl_memberships)
        self.obs_data = np.array(cl_data)
        
        WSS = self._getWSS()
        TSS = self._getTSS()
        self.BSS = TSS - WSS