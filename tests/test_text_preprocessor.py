import unittest
from multimodal_preprocessing.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextPreprocessor(max_features=10)
        self.texts = ["I love machine learning!", "Data science is the future."]

    def test_lemmatize(self):
        lemmatized = self.processor.lemmatize(self.texts[0])
        self.assertIsInstance(lemmatized, str)
        self.assertNotIn("love", lemmatized)  # Example assertion

    def test_preprocess(self):
        features = self.processor.preprocess(self.texts)
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertTrue(features.shape[1] <= 10)

if __name__ == "__main__":
    unittest.main()
