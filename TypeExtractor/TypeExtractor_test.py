from TypeExtractor import TypeExtractor
from pandas.testing import assert_frame_equal
import pandas as pd
import unittest

df = pd.read_csv('https://datahub.io/machine-learning/bank-marketing/r/bank-marketing.csv')
df[['V5', 'V7', 'V8']] = df[['V5', 'V7', 'V8']].applymap(lambda x: True if x=='yes' else False)

class MovingWindowTest(unittest.TestCase):
    def test_select_numeric_only(self):
        df_test = TypeExtractor('number', True).fit_transform(df)
        df_val = df.select_dtypes(include='number')
        assert_frame_equal(df_test, df_val)
    
    def test_exclude_object_and_bool(self):
        df_test = TypeExtractor(['object', bool], False).fit_transform(df)
        df_val = df.select_dtypes(exclude=['object', bool])
        assert_frame_equal(df_test, df_val)
    
    def test_include_dtype_non_existing_in_the_df(self):
        df_test = TypeExtractor('timedelta', True).fit_transform(df)
        df_val = df.select_dtypes(include='timedelta')
        assert_frame_equal(df_test, df_val)

    def test_include_multiple_dtypes_including_one_non_existing_in_the_df(self):
        df_test = TypeExtractor(['object', bool, 'timedelta'], True).fit_transform(df)
        df_val = df.select_dtypes(include=['object', bool, 'timedelta'])
        assert_frame_equal(df_test, df_val)
        
if __name__ == '__main__':
    unittest.main()