import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin

class MovingWindow(BaseEstimator, TransformerMixin):

    @staticmethod
    def moving_window_3d(arr, size):
        N = arr.shape
        s = arr.strides
        return as_strided(arr, shape=(N[0] - size + 1,  size, N[1]), strides=(s[0], s[0], s[1]))
    
    def __init__(self, predictor_window: int, target_window: int):
        self._predictor = predictor_window
        self._target = target_window
    
    def fit(self, X: np.ndarray, y=None): 
        return self
    
    def transform(self, X: np.ndarray, y=None): 
        X = self.moving_window_3d(X[:-self._target], self._predictor)
        return np.flip(X,1)
