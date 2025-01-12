from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NumericPreprocessor:
    def __init__(self, scaler_type='standard'):
        """
        Initialize the NumericPreprocessor with a specified scaler type.
        Options: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
        """
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    def preprocess(self, numeric_data):
        """
        Scale and normalize numeric data using the chosen scaler.
        """
        return self.scaler.fit_transform(numeric_data)
