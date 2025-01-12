import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.nlp = spacy.load("en_core_web_sm")
        # Add custom stopwords
        self.nlp.Defaults.stop_words |= {"love"}  # Add 'love' as a stopword

    def lemmatize(self, text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])


    def preprocess(self, texts):
        """
        Perform lemmatization and vectorization on a list of texts.
        """
        lemmatized_texts = [self.lemmatize(text) for text in texts]
        return self.vectorizer.fit_transform(lemmatized_texts).toarray()


    ####
    