import pandas as pd
from imblearn.over_sampling import SMOTE

class VideoPreprocessor:
    def __init__(self, balance_classes=False):
        """
        Initialize the VideoPreprocessor with an option to balance classes.
        """
        self.balance_classes = balance_classes

    def handle_missing(self, video_data, method='mean'):
        """
        Handle missing values in the video data by filling them with the mean, median, or a constant value.
        """
        if method == 'mean':
            return video_data.fillna(video_data.mean())
        elif method == 'median':
            return video_data.fillna(video_data.median())
        else:
            return video_data.fillna(0)

    def balance_data(self, features, labels):
        """
        Balance the dataset using SMOTE (Synthetic Minority Oversampling Technique).
        """
        smote = SMOTE()
        return smote.fit_resample(features, labels)

    def preprocess(self, video_data, labels=None):
        """
        Preprocess video data by handling missing values and optionally balancing the dataset.
        """
        processed_data = self.handle_missing(video_data)
        if self.balance_classes and labels is not None:
            processed_data, labels = self.balance_data(processed_data, labels)
        return processed_data, labels
