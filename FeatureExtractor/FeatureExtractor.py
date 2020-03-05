from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_names: list):
        self._column_names = column_names
        
    def fit(self, X: pd.DataFrame, y=None):
        return self 
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if isinstance(X[self._column_names], pd.core.frame.DataFrame):
            return X[self._column_names]
        else:
            return X[self._column_names].to_frame()