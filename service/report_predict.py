"""
Module for generating model performance reports and category predictions.

This module provides functions to compute comprehensive metrics and visualizations
for training and test sets, and a helper function for predicting message categories.
"""

import logging

import matplotlib.pyplot as plt  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import seaborn as sns  # pylint: disable=import-error
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, accuracy_score
)  # pylint: disable=import-error

# Import the relevant objects from pipeline.py using relative import
from .pipeline import get_model, process_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _compute_metrics(model, x, y):
    """
    Compute prediction metrics for a dataset.

    Returns:
        report_df: DataFrame of the classification report.
        conf: Confusion matrix.
        accuracy: Accuracy score.
        auc_val: AUC value if binary classification, else None.
        roc_data: Tuple of (fpr, tpr) if binary classification, else None.
    """
    y_pred = model.predict(x)
    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
    conf = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    auc_val = None
    roc_data = None

    # Check if model is a pipeline with a classifier under key 'clf'
    clf = model.named_steps['clf'] if hasattr(model, "named_steps") and 'clf' in model.named_steps else model

    if hasattr(clf, 'classes_') and len(clf.classes_) == 2:
        probs = model.predict_proba(x)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        auc_val = auc(fpr, tpr)
        roc_data = (fpr, tpr)
    return report_df, conf, accuracy, auc_val, roc_data


def _plot_figure(plt_obj, title, xlabel, ylabel, save_plots, filename):
    """
    Save the figure to file if save_plots is True; otherwise, display it.
    """
    plt_obj.title(title)
    plt_obj.xlabel(xlabel)
    plt_obj.ylabel(ylabel)
    if save_plots:
        plt_obj.savefig(filename)
        plt_obj.close()
    else:
        plt_obj.show()


def generate_full_model_report(model, data, save_plots=False):
    """
    Generates and logs a comprehensive report of model performance on both training and test sets.

    The data argument should be a dictionary containing:
        - x_train, y_train, x_test, y_test

    Returns:
        A dictionary with detailed metrics for further analysis.
    """
    logger.info("Generating full model report...")

    # Compute metrics for training data
    train_report_df, train_conf, train_accuracy, auc_train, roc_train = _compute_metrics(
        model, data["x_train"], data["y_train"]
    )

    # Compute metrics for test data
    test_report_df, test_conf, test_accuracy, auc_test, roc_test = _compute_metrics(
        model, data["x_test"], data["y_test"]
    )

    # Retrieve the classifier from a pipeline if available.
    clf = model.named_steps['clf'] if hasattr(model, "named_steps") and 'clf' in model.named_steps else model

    # Log training metrics using f-string formatting
    logger.info("========== TRAINING METRICS ==========")
    logger.info(f"Overall Accuracy: {train_accuracy:.4f}")
    if auc_train is not None:
        logger.info(f"ROC AUC: {auc_train:.4f}")
    logger.info(f"Classification Report (Train):\n{train_report_df}")
    if hasattr(clf, 'classes_'):
        logger.info(f"Confusion Matrix (Train) - Classes: {clf.classes_}")
    logger.info(f"\n{train_conf}")

    # Log testing metrics
    logger.info("========== TESTING METRICS ==========")
    logger.info(f"Overall Accuracy: {test_accuracy:.4f}")
    if auc_test is not None:
        logger.info(f"ROC AUC: {auc_test:.4f}")
    logger.info(f"Classification Report (Test):\n{test_report_df}")
    if hasattr(clf, 'classes_'):
        logger.info(f"Confusion Matrix (Test) - Classes: {clf.classes_}")
    logger.info(f"\n{test_conf}")

    # Plot confusion matrices for training and testing data
    plt.figure(figsize=(6, 5))
    sns.heatmap(train_conf, annot=True, fmt="d", cmap="Blues")
    plt.title("Training Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_plots:
        plt.savefig("train_confusion_matrix.png")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(test_conf, annot=True, fmt="d", cmap="Blues")
    plt.title("Testing Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_plots:
        plt.savefig("test_confusion_matrix.png")
        plt.close()
    else:
        plt.show()

    # Plot ROC curves (only if both training and test ROC data exist)
    if roc_train is not None and roc_test is not None:
        fpr_train, tpr_train = roc_train
        fpr_test, tpr_test = roc_test
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.4f})")
        plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {auc_test:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        if save_plots:
            plt.savefig("roc_curve.png")
            plt.close()
        else:
            plt.show()

    # Plot bar chart for per-class metrics (precision, recall, f1-score)
    aggregate_rows = ["accuracy", "macro avg", "weighted avg"]
    class_labels = [label for label in test_report_df.index if label not in aggregate_rows]
    if class_labels:
        metrics = test_report_df.loc[class_labels, ["precision", "recall", "f1-score"]]
        metrics.plot(kind="bar", figsize=(10, 6))
        plt.title("Classification Metrics by Class (Test)")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend(loc="lower right")
        if save_plots:
            plt.savefig("class_metrics.png")
            plt.close()
        else:
            plt.show()

    return {
        "train": {
            "accuracy": train_accuracy,
            "auc": auc_train,
            "report_df": train_report_df,
            "confusion_matrix": train_conf,
            "roc_data": roc_train,
        },
        "test": {
            "accuracy": test_accuracy,
            "auc": auc_test,
            "report_df": test_report_df,
            "confusion_matrix": test_conf,
            "roc_data": roc_test,
        }
    }


def predict_category(message, classifier_type, top_n=3):
    """
    Preprocesses the input message and predicts its category using the specified classifier.

    If the model supports predict_proba and has multiple classes, returns the top 'top_n'
    predicted categories with their probabilities. Otherwise, returns a single label.
    """
    logger.info("Predicting category for a new message...")
    model = get_model(classifier_type)
    processed_message = process_text(message)  # minimal_preprocess + spaCy + NER

    clf = model.named_steps['clf'] if hasattr(model, "named_steps") and 'clf' in model.named_steps else model

    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([processed_message])[0]
        classes = clf.classes_
        sorted_indices = np.argsort(probs)[::-1]
        top_n = min(top_n, len(classes))
        top_indices = sorted_indices[:top_n]
        predictions = [(classes[i], probs[i]) for i in top_indices]
        return predictions
    category = model.predict([processed_message])[0]
    return [(category, 1.0)]
