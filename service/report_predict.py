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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# Import the relevant objects from pipeline.py using relative import
from .pipeline import (
    get_model,
    process_text,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _compute_metrics(model, x, y):
    """
    Compute prediction metrics for a dataset, supporting multi-class classification.

    Returns:
        report_df: DataFrame of the classification report.
        conf: Confusion matrix.
        accuracy: Accuracy score.
        auc_vals: Dictionary of AUC values for each class (One-vs-Rest strategy).
        roc_data: Dictionary of (fpr, tpr) for each class.
    """
    y_pred = model.predict(x)
    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
    conf = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    clf = model.named_steps['clf'] if hasattr(model, "named_steps") and 'clf' in model.named_steps else model

    auc_vals = None
    roc_data = None

    if hasattr(clf, 'predict_proba'):
        y_prob = model.predict_proba(x)
        n_classes = y_prob.shape[1]
        auc_vals = {}
        roc_data = {}

        for i in range(n_classes):
            y_bin = label_binarize(y, classes=np.unique(y))  # Binarizza y
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])  # Usa la colonna i-esima

            auc_vals[i] = auc(fpr, tpr)
            roc_data[i] = (fpr, tpr)

    return report_df, conf, accuracy, auc_vals, roc_data


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
    Generates a report for model performance on training and test sets, supporting multi-class ROC curves.
    """
    logger.info("Generating full model report...")

    train_report_df, train_conf, train_accuracy, auc_train, roc_train = _compute_metrics(model, data["x_train"],
                                                                                         data["y_train"])
    test_report_df, test_conf, test_accuracy, auc_test, roc_test = _compute_metrics(model, data["x_test"],
                                                                                    data["y_test"])

    clf = model.named_steps['clf'] if hasattr(model, "named_steps") and 'clf' in model.named_steps else model

    logger.info("========== TRAINING METRICS ==========")
    logger.info("Overall Accuracy: %.4f", train_accuracy)
    if auc_train is not None:
        logger.info("ROC AUC per class: %s", auc_train)
    else: logger.info("ROC AUC per class: None")
    logger.info("Classification Report (Train):\n%s", train_report_df)
    logger.info("Confusion Matrix (Train):\n%s", train_conf)

    logger.info("========== TESTING METRICS ==========")
    logger.info("Overall Accuracy: %.4f", test_accuracy)
    if auc_test is not None:
        logger.info("ROC AUC per class: %s", auc_test)
    logger.info("Classification Report (Test):\n%s", test_report_df)
    logger.info("Confusion Matrix (Test):\n%s", test_conf)

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

    if roc_test is not None:
        plt.figure(figsize=(8, 6))
        for class_idx, (fpr, tpr) in roc_test.items():
            plt.plot(fpr, tpr, label=f"Class {class_idx} (AUC = {auc_test[class_idx]:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-Class ROC Curve")
        plt.legend(loc="lower right")
        if save_plots:
            plt.savefig("roc_curve.png")
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
    Preprocesses the input message and predicts its category using the specified
    classifier.

    If the model supports predict_proba and has multiple classes, returns the
    top 'top_n' predicted categories with their probabilities. Otherwise, returns
    a single label.
    """
    logger.info("Predicting category for a new message...")
    model = get_model(classifier_type)
    processed_message = process_text(message)  # minimal_preprocess + spaCy + NER

    clf = (model.named_steps['clf'] if hasattr(model, "named_steps") and
           'clf' in model.named_steps else model)

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


def evaluate_model_with_kfold(model, x_processed, y, save_plots=False):
    """
    Evaluate the provided model using 5‑fold StratifiedKFold.

    For each fold, this function calls generate_full_model_report to obtain metrics,
    then aggregates key statistics:
      - Average accuracy and AUC for train and test sets.
      - Average confusion matrix for train and test sets.
      - Average classification report (averaging numeric metrics) for train and test sets.
      - Overfitting rating: the difference between training and test accuracy per fold.

    Additionally, it produces a bar graph showing the overfitting differences per fold.

    Args:
        model: The trained model to evaluate.
        x_processed: Feature data (pandas Series or numpy array).
        y: Target labels.
        save_plots: Boolean flag passed to generate_full_model_report.

    Returns:
        final_avg_report: A dictionary containing the averaged metrics and overfitting stats.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store metrics per fold.
    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    train_conf_matrices = []
    test_conf_matrices = []
    train_reports = []  # classification report DataFrames for train
    test_reports = []   # classification report DataFrames for test
    overfitting_diffs = []  # train accuracy minus test accuracy per fold

    for fold, (train_index, test_index) in enumerate(skf.split(x_processed, y), start=1):
        if isinstance(x_processed, pd.Series):
            x_train_fold = x_processed.iloc[train_index]
            x_test_fold = x_processed.iloc[test_index]
        else:
            x_train_fold = x_processed[train_index]
            x_test_fold = x_processed[test_index]
        if isinstance(y, pd.Series):
            y_train_fold = y.iloc[train_index]
            y_test_fold = y.iloc[test_index]
        else:
            y_train_fold = y[train_index]
            y_test_fold = y[test_index]

        data_dict = {
            "x_train": x_train_fold,
            "y_train": y_train_fold,
            "x_test": x_test_fold,
            "y_test": y_test_fold
        }

        # Generate report for the fold.
        report_fold = generate_full_model_report(model, data_dict, save_plots=save_plots)

        # Collect accuracies and AUCs.
        train_acc = report_fold["train"]["accuracy"]
        test_acc = report_fold["test"]["accuracy"]
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_aucs.append(report_fold["train"]["auc"])
        test_aucs.append(report_fold["test"]["auc"])
        overfitting_diffs.append(train_acc - test_acc)

        # Collect confusion matrices.
        train_conf_matrices.append(report_fold["train"]["confusion_matrix"])
        test_conf_matrices.append(report_fold["test"]["confusion_matrix"])

        # Collect classification reports (DataFrames).
        train_reports.append(report_fold["train"]["report_df"])
        test_reports.append(report_fold["test"]["report_df"])

    # Compute averages (ignoring None values in AUC lists).
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)

    # Estrai i valori AUC dalle matrici auc_train e auc_test
    train_auc_values = [auc for auc in train_aucs if isinstance(auc, (int, float))]
    test_auc_values = [auc for auc in train_aucs if isinstance(auc, (int, float))]

    # Calcola la media solo sui valori numerici.
    avg_train_auc = np.mean(train_auc_values) if train_auc_values else None
    avg_test_auc = np.mean(test_auc_values) if test_auc_values else None

    # Average confusion matrix (as a numpy array).
    avg_train_conf = np.mean(np.array(train_conf_matrices), axis=0)
    avg_test_conf = np.mean(np.array(test_conf_matrices), axis=0)#

    # Average classification reports: concatenate DataFrames and group by index.
    avg_train_report = pd.concat(train_reports).groupby(level=0).mean()
    avg_test_report = pd.concat(test_reports).groupby(level=0).mean()

    avg_overfitting_diff = np.mean(overfitting_diffs)

    final_avg_report = {
        "train": {
            "avg_accuracy": avg_train_accuracy,
            "avg_auc": avg_train_auc,
            "avg_confusion_matrix": avg_train_conf,
            "avg_classification_report": avg_train_report
        },
        "test": {
            "avg_accuracy": avg_test_accuracy,
            "avg_auc": avg_test_auc,
            "avg_confusion_matrix": avg_test_conf,
            "avg_classification_report": avg_test_report
        },
        "overfitting": {
            "fold_differences": overfitting_diffs,
            "avg_difference": avg_overfitting_diff
        }
    }

    # Plot overfitting differences per fold.
    plt.figure(figsize=(8, 6))
    fold_numbers = np.arange(1, len(overfitting_diffs) + 1)
    plt.bar(fold_numbers, overfitting_diffs, color='salmon')
    plt.xlabel("Fold Number")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.title("Overfitting Rating per Fold")
    plt.axhline(y=avg_overfitting_diff, color='blue', linestyle='--', label="Average Overfitting")
    plt.legend()
    if save_plots:
        plt.savefig("overfitting_rating.png")
        plt.close()
    else:
        plt.show()

    logger.info("Final average evaluation: %s", final_avg_report)
    return final_avg_report
