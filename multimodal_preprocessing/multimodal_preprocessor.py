import numpy as np
from .text_preprocessor import TextPreprocessor
from .numeric_preprocessor import NumericPreprocessor
from .video_preprocessor import VideoPreprocessor

class MultimodalPreprocessor:
    def __init__(self, text_max_features=5000, numeric_scaler_type='standard', video_balance=False):
        """
        Initialize the MultimodalPreprocessor with modality-specific preprocessors.
        """
        self.text_preprocessor = TextPreprocessor(max_features=text_max_features)
        self.numeric_preprocessor = NumericPreprocessor(scaler_type=numeric_scaler_type)
        self.video_preprocessor = VideoPreprocessor(balance_classes=video_balance)

    def preprocess_text(self, texts):
        """
        Preprocess text data using the TextPreprocessor.
        """
        return self.text_preprocessor.preprocess(texts)

    def preprocess_numeric(self, numeric_data):
        """
        Preprocess numeric data using the NumericPreprocessor.
        """
        return self.numeric_preprocessor.preprocess(numeric_data)

    def preprocess_video(self, video_data, labels=None):
        """
        Preprocess video data using the VideoPreprocessor.
        """
        return self.video_preprocessor.preprocess(video_data, labels)

    def preprocess_all(self, texts, numeric_data, video_data, labels=None):
        """
        Preprocess all modalities (text, numeric, and video) and combine the features.
        """
        text_features = self.preprocess_text(texts)
        numeric_features = self.preprocess_numeric(numeric_data)
        video_features, labels = self.preprocess_video(video_data, labels)
        combined_features = np.concatenate([text_features, numeric_features, video_features], axis=1)
        return combined_features, labels
