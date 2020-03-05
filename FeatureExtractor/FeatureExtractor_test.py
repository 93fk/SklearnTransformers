import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.datasets import load_breast_cancer
from FeatureExtractor import FeatureExtractor

import unittest

bc = load_breast_cancer()
df = pd.DataFrame(data=bc.data, columns=bc.feature_names)

class FeatureExtractorTest(unittest.TestCase):
    def test_get_3_exisitng_columns(self):
        col_names = ['mean radius', 'mean texture', 'mean perimeter']
        df_test = FeatureExtractor(col_names).fit_transform(df)
        df_val = df[col_names]
        assert_frame_equal(df_test, df_val)

    def test_get_1_existing_column(self):
        col_name = 'mean radius'
        df_test = FeatureExtractor(col_name).fit_transform((df))
        df_val = df[[col_name]]
        assert_frame_equal(df_test, df_val)

if __name__ == '__main__':
    unittest.main()