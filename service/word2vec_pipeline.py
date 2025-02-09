import os
from typing import Tuple, Dict, Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
import psutil
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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


def evaluate_word2vec_model_kfold(
        classifier_type: ClassifierType,
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5
) -> Dict[str, Any]:
    """
    Evaluate the Word2Vec-based model using 5-fold cross-validation.
    For each fold, the model is cloned (from get_word2vec_model), trained, and evaluated.
    The function prints training and testing accuracies, confusion matrix, and classification report for each fold,
    and then prints the average metrics over all folds.

    Parameters:
        classifier_type (ClassifierType): The classifier type (e.g., SVM, NAIVE_BAYES).
        X (Union[pd.Series, np.ndarray]): Input data (preprocessed text).
        y (Union[pd.Series, np.ndarray]): Associated labels.
        n_splits (int): Number of folds (default 5).

    Returns:
        Dict[str, Any]: A dictionary containing aggregated metrics.
    """
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    unique_labels = np.unique(y)
    train_accs, test_accs, conf_mats, reports = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"=== Fold {fold} ===")
        X_train = X.iloc[train_idx] if isinstance(X, pd.Series) else X[train_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.Series) else X[test_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]

        # Clone and train model
        model = clone(get_word2vec_model(classifier_type))
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics calculation
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        conf_mat = confusion_matrix(y_test, y_pred_test, labels=unique_labels)
        report = classification_report(y_test, y_pred_test, labels=unique_labels, output_dict=True, zero_division=0)

        # Print metrics for current fold
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print("Confusion Matrix:\n", conf_mat)
        print("Classification Report:")
        for label, metrics in report.items():
            print(f"  {label}: {metrics}")
        print("-" * 40)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        conf_mats.append(conf_mat)
        reports.append(pd.DataFrame(report).transpose())

    # Aggregate metrics over folds
    avg_train_acc = np.mean(train_accs)
    avg_test_acc = np.mean(test_accs)
    avg_conf_mat = np.mean(conf_mats, axis=0)
    avg_report = pd.concat(reports).groupby(level=0).mean()

    print("=== Average Results over 5 Folds ===")
    print(f"Average Train Accuracy: {avg_train_acc:.4f}")
    print(f"Average Test Accuracy:  {avg_test_acc:.4f}")
    print("Average Confusion Matrix:\n", avg_conf_mat)
    print("Average Classification Report:\n", avg_report)

    return {
        "train": {
            "accuracy": avg_train_acc,
            "confusion_matrix": avg_conf_mat,
            "classification_report": avg_report
        },
        "test": {
            "accuracy": avg_test_acc
        }
    }
