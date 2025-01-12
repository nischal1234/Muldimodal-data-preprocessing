import unittest
import numpy as np
from multimodal_preprocessing.numeric_preprocessor import NumericPreprocessor

class TestNumericPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = NumericPreprocessor(scaler_type='minmax')
        self.numeric_data = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_preprocess(self):
        scaled_data = self.processor.preprocess(self.numeric_data)
        self.assertEqual(scaled_data.shape, self.numeric_data.shape)
        self.assertTrue(np.all(scaled_data >= 0) and np.all(scaled_data <= 1))

if __name__ == "__main__":
    unittest.main()
