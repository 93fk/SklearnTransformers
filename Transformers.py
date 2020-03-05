import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin

class MovingWindowLSTM(BaseEstimator, TransformerMixin):
    """Returns 3D numpy.ndarray ready to fedd into LSTM like models.

    This transformer takes a snap of a 2D array, reverses its order (so the most
    recent observations are at the start of a subarray) and appends it to a 3D array.

    Parameters
    ----------
    predictor_window: int, required
        number of prevoius steps to use in the prediciton.

    target_window: int, required
        number of next steps to predict.
    ----------

    Use example:
    >>> import numpy as np
    >>> from Transformers import MovingWindowLSTM
    >>> array = np.arange(12).reshape(6,2)
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 8,  9],
           [10, 11]])
    >>> new_array = MovingWindowLSTM(3,2)
    array([[[4, 5],
            [2, 3],
            [0, 1]],

           [[6, 7],
            [4, 5],
            [2, 3]]])
    """

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