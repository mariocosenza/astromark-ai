import os
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import psutil
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from service.pipeline import ClassifierType, TRAINED_DIR, logger, X_processed, y, process_text


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that learns Word2Vec embeddings from the training corpus
    and transforms each document into the average of its word vectors.
    """

    def __init__(self, vector_size: int = 100, min_count: int = 1, workers: int = 1):
        self.vector_size = vector_size
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        # X is expected to be an iterable of preprocessed text strings.
        tokenized_texts = [text.split() for text in X]
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=42
        )
        return self

    def transform(self, X):
        tokenized_texts = [text.split() for text in X]
        features = []
        for tokens in tokenized_texts:
            # Compute the average vector for the document.
            word_vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
            if word_vectors:
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                avg_vector = np.zeros(self.vector_size)
            features.append(avg_vector)
        return np.array(features)


def build_word2vec_pipeline(classifier_type: ClassifierType) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Build a machine learning pipeline that uses Word2Vec embeddings for text representation.
    Returns a tuple of (pipeline, parameter grid) for grid search.
    """
    # Use all available cores for training Word2Vec.
    num_workers = psutil.cpu_count(logical=True)
    w2v = Word2VecVectorizer(vector_size=100, min_count=1, workers=num_workers)

    if classifier_type == ClassifierType.NAIVE_BAYES:
        # Note: MultinomialNB is generally not ideal for continuous (and possibly negative)
        # features produced by Word2Vec. You might consider a GaussianNB or similar instead.
        classifier = MultinomialNB()
        pipeline = Pipeline([
            ('w2v', w2v),
            ('clf', classifier)
        ])
        param_grid = {
            'w2v__vector_size': [50, 100, 200],
            'clf__alpha': [1.0, 1.5, 2.0]
        }
    elif classifier_type == ClassifierType.SVM:
        classifier = SVC(probability=True, kernel='linear', random_state=42)
        pipeline = Pipeline([
            ('w2v', w2v),
            ('clf', classifier)
        ])
        param_grid = {
            'w2v__vector_size': [50, 100, 200],
            'clf__C': [0.1, 0.5, 1.0, 2.0]
        }
    else:
        raise ValueError("Unsupported classifier type for Word2Vec pipeline.")
    return pipeline, param_grid


# Define separate model file paths for the Word2Vec-based models.
W2V_MODEL_PATHS = {
    "naive_bayes": os.path.join(TRAINED_DIR, "trained_model_nb_w2v.pkl"),
    "svm": os.path.join(TRAINED_DIR, "trained_model_svm_w2v.pkl")
}


def save_word2vec_model(model: Pipeline, classifier_type: ClassifierType) -> None:
    """
    Save the trained Word2Vec-based model to disk.
    """
    path = W2V_MODEL_PATHS[classifier_type.value]
    joblib.dump(model, path)
    logger.info("Word2Vec model saved to %s.", path)


def load_word2vec_model(classifier_type: ClassifierType) -> Optional[Pipeline]:
    """
    Load a previously saved Word2Vec-based model from disk, if it exists.
    """
    path = W2V_MODEL_PATHS[classifier_type.value]
    if os.path.exists(path):
        logger.info("Loading saved Word2Vec model from %s...", path)
        return joblib.load(path)
    return None


def get_word2vec_model(classifier_type: ClassifierType) -> Pipeline:
    """
    Load a pre-trained Word2Vec-based model if available; otherwise, train via grid search and save the model.
    Returns the model.
    """
    model = load_word2vec_model(classifier_type)
    if model is None:
        logger.info("No saved Word2Vec model found for %s. Training a new one...", classifier_type.value)
        pipeline, param_grid = build_word2vec_pipeline(classifier_type)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=kf, n_jobs=-1, verbose=0)
        grid.fit(X_processed, y)
        logger.info("Best parameters for Word2Vec pipeline: %s", grid.best_params_)
        model = grid.best_estimator_
        save_word2vec_model(model, classifier_type)
    else:
        logger.info("Using saved Word2Vec model for %s.", classifier_type.value)
    return model


def classify_with_word2vec(text: str, classifier_type: ClassifierType = ClassifierType.SVM) -> Any:
    """
    Given a raw text input, perform minimal preprocessing, obtain the Word2Vec-based model,
    and return the predicted category.
    """
    processed_text = process_text(text)
    model = get_word2vec_model(classifier_type)
    prediction = model.predict([processed_text])
    logger.info("Prediction for the input text using Word2Vec pipeline: %s", prediction[0])
    return prediction[0]
