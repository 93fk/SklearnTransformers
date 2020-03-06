import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Returns selected columns from a pandas DataFrame object.

    Parameters
    ----------
    column_names: list, string; required
        columns to be selected.
    ----------

    Use example:
    import pandas as pd

    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    >>> print(df.to_markdown())
    |    |   A |   B |   C |
    |---:|----:|----:|----:|
    |  0 |   1 |   2 |   3 |
    |  1 |   4 |   5 |   6 |
    |  2 |   7 |   8 |   9 |
    >>> new_df = FeatureExtractor(['A', 'C']).fit_transform(df)
    >>> print(new_df.to_markdown())
    |    |   A |   C |
    |---:|----:|----:|
    |  0 |   1 |   3 |
    |  1 |   4 |   6 |
    |  2 |   7 |   9 |
    """
    def __init__(self, column_names: list):
        self._column_names = column_names
        
    def fit(self, X: pd.DataFrame, y=None):
        return self 
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if isinstance(X[self._column_names], pd.core.frame.DataFrame):
            return X[self._column_names]
        else:
            return X[self._column_names].to_frame()

class TypeExtractor(BaseEstimator, TransformerMixin):
    """This Transformer acts as a Sklearn.Pipeline wrapper for pandas.select_dtypes() method

    Parameters
    ----------
    col_type: string, list-like; required
        name of dtype(s) to be acted on.

    include: bool: required
        whether or not selected dtypes should be inlcuded or excluded.
    ----------

    Use example:
    import pandas as pd

    >>> df = pd.DataFrame([[1, 'X', True], [2, 'Y', True], [3, 'Z', False]], columns=['A', 'B', 'C'])
    >>> print(df.to_markdown())
    |    |   A | B   | C     |
    |---:|----:|:----|:------|
    |  0 |   1 | X   | True  |
    |  1 |   2 | Y   | True  |
    |  2 |   3 | Z   | False |
    >>> new_df = TypeExtactor('number', include=True).fit_transform(df)
    >>> print(new_df.to_markdown())
    |    |   A |
    |---:|----:|
    |  0 |   1 |
    |  1 |   2 |
    |  2 |   3 |
    """
    def __init__(self, col_type: str, include: bool=True):
        self._col_type = col_type
        self._include = include
        
    def fit(self, X: pd.DataFrame, y=None):
        if self._include:
            self._columns = X.select_dtypes(include=self._col_type).columns._data.tolist()
        else:
            self._columns = X.select_dtypes(exclude=self._col_type).columns._data.tolist()
        return self 
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if isinstance(X[self._columns], pd.core.frame.DataFrame):
            return X[self._columns]
        else:
            return X[self._columns].to_frame()

class MovingWindowLSTM(BaseEstimator, TransformerMixin):
    """Returns 3D numpy.ndarray ready to fedd into LSTM like models.

    This transformer takes a snap of a 2D array, reverses its order (so the most
    recent observations are at the start of a subarray) and appends it to a 3D array.

    Parameters
    ----------
    predictor_window: int; required
        number of prevoius steps to use in the prediciton.

    target_window: int; required
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
    >>> new_array = MovingWindowLSTM(3,2).fit_transform(array)
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
