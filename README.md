---

# **Multimodal Preprocessing Library**

A Python library designed for efficient preprocessing of multimodal data, including **text**, **numeric**, and **video** modalities. This library simplifies the preprocessing pipeline, enabling seamless integration into machine learning workflows.

---

## **Features**
- **Text Preprocessing**:
  - Lemmatization using SpaCy.
  - Stopword removal.
  - TF-IDF vectorization with customizable vocabulary size.
- **Numeric Data Preprocessing**:
  - Scaling and normalization using StandardScaler or MinMaxScaler.
- **Video Data Preprocessing**:
  - Handling missing values with mean, median, or constant values.
  - Balancing imbalanced datasets using SMOTE (Synthetic Minority Oversampling Technique).
- **Multimodal Preprocessing**:
  - Combines preprocessed features from multiple modalities into a unified feature matrix.

---

## **Installation**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/multimodal-preprocessing.git
cd multimodal-preprocessing
```

### **Step 2: Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **Step 3: Download SpaCy Language Model**
Ensure the SpaCy language model is installed:
```bash
python -m spacy download en_core_web_sm
```

---

## **Usage**
### **Quick Start**
1. **Import the Library**:
   ```python
   from multimodal_preprocessing.multimodal_preprocessor import MultimodalPreprocessor
   ```

2. **Initialize the Preprocessor**:
   ```python
   preprocessor = MultimodalPreprocessor(
       text_max_features=5000,
       numeric_scaler_type='minmax',
       video_balance=True
   )
   ```

3. **Preprocess Data**:
   ```python
   texts = ["I love data science!", "Machine learning is amazing.", "Deep learning powers AI."]
   numeric_data = [[1.0, 20.0], [2.0, 30.0], [3.0, 40.0]]
   video_data = [[0.5, None], [0.8, 1.2], [None, 0.9]]
   labels = [0, 1, 1]

   # Preprocess individual modalities
   text_features = preprocessor.preprocess_text(texts)
   numeric_features = preprocessor.preprocess_numeric(numeric_data)
   video_features, balanced_labels = preprocessor.preprocess_video(video_data, labels)

   # Combine all modalities
   combined_features, combined_labels = preprocessor.preprocess_all(texts, numeric_data, video_data, labels)

   print("Text Features Shape:", text_features.shape)
   print("Combined Features Shape:", combined_features.shape)
   ```

---

## **Modules**
### **1. Text Preprocessor**
Lemmatizes and vectorizes text data.
```python
from multimodal_preprocessing.text_preprocessor import TextPreprocessor
processor = TextPreprocessor(max_features=5000)
processed_text = processor.preprocess(["Example sentence for text preprocessing."])
```

### **2. Numeric Preprocessor**
Scales numeric data using StandardScaler or MinMaxScaler.
```python
from multimodal_preprocessing.numeric_preprocessor import NumericPreprocessor
processor = NumericPreprocessor(scaler_type='standard')
scaled_data = processor.preprocess([[1.0, 2.0], [3.0, 4.0]])
```

### **3. Video Preprocessor**
Handles missing values and balances datasets.
```python
from multimodal_preprocessing.video_preprocessor import VideoPreprocessor
processor = VideoPreprocessor(balance_classes=True)
processed_video, labels = processor.preprocess([[0.5, None], [0.8, 1.2]], [0, 1])
```

### **4. Multimodal Preprocessor**
Combines features from all modalities.
```python
from multimodal_preprocessing.multimodal_preprocessor import MultimodalPreprocessor
processor = MultimodalPreprocessor()
combined_features, labels = processor.preprocess_all(texts, numeric_data, video_data, labels)
```

---

## **Testing**
Run the unit tests to verify the functionality:
```bash
python -m unittest discover tests
```

For `pytest`:
```bash
pytest
```

---

## **Dependencies**
- **Python**: >= 3.7
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `spacy`
  - `imbalanced-learn`

---

## **Contributing**
Contributions are welcome! If you'd like to improve this library:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request.

---

## **License**
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## **Contact**
For questions or support, feel free to contact:
- **Name**: Nischal Mandal
- **Email**: nishchalmandal@gmail.com
- **GitHub**: https://github.com/nischal1234

---
