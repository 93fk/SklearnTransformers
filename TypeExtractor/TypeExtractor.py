from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TypeExtractor(BaseEstimator, TransformerMixin):
    
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