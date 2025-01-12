import unittest
import numpy as np
import pandas as pd
from multimodal_preprocessing.multimodal_preprocessor import MultimodalPreprocessor

class TestMultimodalPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = MultimodalPreprocessor()
        self.texts = ["I love data science.", "AI is fascinating."]
        self.numeric_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.video_data = pd.DataFrame([[0.5, None], [1.0, 2.0]])
        self.labels = [0, 1]

    def test_preprocess_all(self):
        combined_features, combined_labels = self.processor.preprocess_all(
            self.texts, self.numeric_data, self.video_data, self.labels
        )
        self.assertIsNotNone(combined_features)
        self.assertEqual(len(combined_labels), len(self.labels))

if __name__ == "__main__":
    unittest.main()
