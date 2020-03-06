from MovingWindow import MovingWindow

import unittest
import numpy as np

array = np.array(([4,5,6],
                  [8,5,7],
                  [1,0,3],
                  [2,9,3],
                  [7,2,9],
                  [4,8,2],
                  [2,2,6],
                  [2,1,1],
                  [9,0,1]))

class MovingWindowTest(unittest.TestCase):
    def test_get_2_predictions_for_3_targets(self):
        MW_2_3 = MovingWindow(2,3)
        test_arr = MW_2_3.fit_transform(array)
        valid_arr = np.array([[[8,5,7],[4,5,6]],
                              [[1,0,3],[8,5,7]],
                              [[2,9,3],[1,0,3]],
                              [[7,2,9],[2,9,3]],
                              [[4,8,2],[7,2,9]]])
        np.testing.assert_array_equal(test_arr.shape, valid_arr.shape)
        np.testing.assert_array_equal(test_arr, valid_arr)

    def test_get_3_predictions_for_2_targets(self):
        MW_3_2 = MovingWindow(3,2)
        test_arr = MW_3_2.fit_transform(array)
        valid_arr = np.array([[[1,0,3],[8,5,7],[4,5,6]],
                              [[2,9,3],[1,0,3],[8,5,7]],
                              [[7,2,9],[2,9,3],[1,0,3]],
                              [[4,8,2],[7,2,9],[2,9,3]],
                              [[2,2,6],[4,8,2],[7,2,9]]])
        np.testing.assert_array_equal(test_arr.shape, valid_arr.shape)
        np.testing.assert_array_equal(test_arr, valid_arr)

    def test_get_5_predictions_for_3_targets(self):
        MW_5_3 = MovingWindow(5,3)
        test_arr = MW_5_3.fit_transform(array)
        valid_arr = np.array([[[7,2,9],[2,9,3],[1,0,3],[8,5,7],[4,5,6]],
                              [[4,8,2],[7,2,9],[2,9,3],[1,0,3],[8,5,7]]])
        np.testing.assert_array_equal(test_arr.shape, valid_arr.shape)
        np.testing.assert_array_equal(test_arr, valid_arr)

if __name__ == '__main__':
    unittest.main()