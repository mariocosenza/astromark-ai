import logging
from typing import Any, Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold

# Import pipeline objects
from .pipeline import (
    process_text,
    get_model,
    ClassifierType,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_category(
        message: str,
        classifier_type: ClassifierType,
        top_n: int = 3
) -> List[Tuple[Any, float]]:
    """
    Preprocess the input message and predict its category using the specified classifier.
    Returns top-N predicted labels and probabilities if available.
    """
    logger.info("Predicting category for a new message...")
    model = get_model(classifier_type)
    processed_message = process_text(message)
    clf = model.named_steps.get('clf', model)  # Use dict-like access if available

    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([processed_message])[0]
        classes = clf.classes_
        sorted_indices = np.argsort(probs)[::-1]
        top_n = min(top_n, len(classes))
        predictions = [(classes[i], probs[i]) for i in sorted_indices[:top_n]]
        return predictions

    # Fallback: predict single label with probability 1.0
    category = model.predict([processed_message])[0]
    return [(category, 1.0)]


def evaluate_model_with_kfold(
        model: Any,
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        save_plots: bool = False,
        folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
) -> Dict[str, Any]:
    """
    Perform K-fold cross-validation on the given trained pipeline.
    For each fold:
      - Clone the pipeline
      - Fit on the training subset and evaluate on train and test data

    If save_plots=True, produces plots of averaged confusion matrices,
    classification metrics, and overfitting differences.
    If `folds` is provided, these splits are used instead of generating new ones.

    Returns a dictionary of averaged metrics across folds.
    """
    unique_labels = np.unique(y)
    if folds is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = list(skf.split(X, y))

    # Containers for fold metrics
    train_accs, test_accs = [], []
    train_confs, test_confs = [], []
    train_reports, test_reports = [], []
    overfit_diffs = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        logger.info("=== Fold %d/%d ===", fold_idx, n_splits)
        X_train = X.iloc[train_idx] if isinstance(X, pd.Series) else X[train_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.Series) else X[test_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        # Train evaluation
        y_pred_train = fold_model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_accs.append(train_acc)
        train_confs.append(confusion_matrix(y_train, y_pred_train, labels=unique_labels))
        train_reports.append(pd.DataFrame(
            classification_report(y_train, y_pred_train, labels=unique_labels, zero_division=0, output_dict=True)
        ).transpose())

        # Test evaluation
        y_pred_test = fold_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_accs.append(test_acc)
        test_confs.append(confusion_matrix(y_test, y_pred_test, labels=unique_labels))
        test_reports.append(pd.DataFrame(
            classification_report(y_test, y_pred_test, labels=unique_labels, zero_division=0, output_dict=True)
        ).transpose())

        overfit_diffs.append(train_acc - test_acc)

    # Compute averages
    avg_train_acc = float(np.mean(train_accs))
    avg_test_acc = float(np.mean(test_accs))
    avg_overfit = float(np.mean(overfit_diffs))
    avg_train_conf = np.mean(train_confs, axis=0)
    avg_test_conf = np.mean(test_confs, axis=0)
    avg_train_report = pd.concat(train_reports).groupby(level=0).mean()
    avg_test_report = pd.concat(test_reports).groupby(level=0).mean()


    results = {
        "train": {
            "accuracy": avg_train_acc,
            "confusion_matrix": avg_train_conf,
            "report_df": avg_train_report
        },
        "test": {
            "accuracy": avg_test_acc,
            "confusion_matrix": avg_test_conf,
            "report_df": avg_test_report
        },
        "overfitting": {
            "differences_per_fold": overfit_diffs,
            "avg_difference": avg_overfit
        }
    }

    plot_kfold_summary(results, classifier_name="MultiClassModel", save=save_plots)
    return results


def finalize_plot(save_plots: bool, filename: Optional[str] = None) -> None:
    """
    Finalize the current plot: if save_plots is True and filename provided,
    save the plot; otherwise, display it.
    """
    plt.tight_layout()
    if save_plots and filename:
        plt.savefig(filename)
        plt.close()
        logger.info("Plot saved to '%s'", filename)
    else:
        plt.show()


def _plot_confusion_matrix(
        conf_mat: np.ndarray,
        title: str,
        save_plots: bool,
        filename: Optional[str] = None
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt="g", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    finalize_plot(save_plots, filename)


def _plot_bar_classification_metrics(
        df_report: pd.DataFrame,
        title: str,
        save_plots: bool,
        filename: Optional[str] = None
) -> None:
    aggregate_rows = {"accuracy", "macro avg", "weighted avg"}
    class_labels = [idx for idx in df_report.index if idx not in aggregate_rows]
    if not class_labels:
        return
    metrics = df_report.loc[class_labels, ["precision", "recall", "f1-score"]]
    metrics.plot(kind="bar", figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    finalize_plot(save_plots, filename)


def _plot_overfitting_bar(
        diffs: List[float],
        classifier_name: str,
        save_plots: bool,
        filename: Optional[str] = None
) -> None:
    fold_numbers = np.arange(1, len(diffs) + 1)
    plt.figure(figsize=(7, 5))
    plt.bar(fold_numbers, diffs, color='orange')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Fold #")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.title(f"{classifier_name} - Overfitting Differences per Fold")
    finalize_plot(save_plots, filename)

def plot_kfold_summary(
        kfold_results: Dict[str, Any],
        classifier_name: str = "MultiClassModel",
        save: bool = False
) -> None:
    """
    Generate summary plots for K-Fold results including:
      - Averaged train/test confusion matrices
      - Bar charts for train/test classification metrics
      - Bar chart for overfitting differences
    """
    _plot_confusion_matrix(
        kfold_results["train"]["confusion_matrix"],
        title=f"{classifier_name} - Train Confusion Matrix (Avg)",
        save_plots=save,
        filename=f"{classifier_name}_train_conf.png"
    )
    _plot_confusion_matrix(
        kfold_results["test"]["confusion_matrix"],
        title=f"{classifier_name} - Test Confusion Matrix (Avg)",
        save_plots=save,
        filename=f"{classifier_name}_test_conf.png"
    )
    _plot_bar_classification_metrics(
        kfold_results["train"]["report_df"],
        title=f"{classifier_name} - Train Metrics (Avg)",
        save_plots=save,
        filename=f"{classifier_name}_train_metrics.png"
    )
    _plot_bar_classification_metrics(
        kfold_results["test"]["report_df"],
        title=f"{classifier_name} - Test Metrics (Avg)",
        save_plots=save,
        filename=f"{classifier_name}_test_metrics.png"
    )
    _plot_overfitting_bar(
        kfold_results["overfitting"]["differences_per_fold"],
        classifier_name,
        save_plots=save,
        filename=f"{classifier_name}_overfitting_bar.png" if save else None
    )


def compare_classifiers_with_kfold(
        X: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        save_plots: bool = False,
        save_data: bool = False
) -> pd.DataFrame:
    """
    Compare Naive Bayes vs. SVM using K‑fold cross-validation on the given data.
    Loads each model via get_model(), evaluates using the same folds,
    and returns a DataFrame summarizing average metrics.
    """
    logger.info("Comparing Naive Bayes vs. SVM with %d-fold cross-validation...", n_splits)
    nb_model = get_model(ClassifierType.NAIVE_BAYES)
    svm_model = get_model(ClassifierType.SVM)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = list(skf.split(X, y))

    logger.info("Evaluating Naive Bayes with KFold...")
    nb_results = evaluate_model_with_kfold(nb_model, X, y, n_splits, shuffle, random_state, save_plots, folds=folds)
    logger.info("Evaluating SVM with KFold...")
    svm_results = evaluate_model_with_kfold(svm_model, X, y, n_splits, shuffle, random_state, save_plots, folds=folds)

    data = {
        "Classifier": ["Naive_Bayes", "SVM"],
        "Train Accuracy": [nb_results["train"]["accuracy"], svm_results["train"]["accuracy"]],
        "Test Accuracy": [nb_results["test"]["accuracy"], svm_results["test"]["accuracy"]],
        "Overfitting (Train-Test)": [
            nb_results["overfitting"]["avg_difference"],
            svm_results["overfitting"]["avg_difference"]
        ]
    }
    df_compare = pd.DataFrame(data)
    logger.info("KFold Comparison:\n%s", df_compare.to_string(index=False))

    if save_data:
        df_compare.to_csv("kfold_comparison.csv", index=False)
        logger.info("Comparison DataFrame saved to 'kfold_comparison.csv'.")

    # Plot test accuracy comparison
    plt.figure(figsize=(6, 4))
    plt.bar(df_compare["Classifier"], df_compare["Test Accuracy"], color=['skyblue', 'coral'])
    plt.title(f"Classifier Comparison - {n_splits}-Fold Test Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate(df_compare["Test Accuracy"]):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
    finalize_plot(save_plots, filename="kfold_test_accuracy_comparison.png" if save_plots else None)

    # Overfitting differences bar chart
    nb_overfit = nb_results["overfitting"]["differences_per_fold"]
    svm_overfit = svm_results["overfitting"]["differences_per_fold"]
    fold_indices = np.arange(n_splits)
    x_positions_nb = fold_indices * 2.0
    x_positions_svm = fold_indices * 2.0 + 0.8

    plt.figure(figsize=(8, 5))
    plt.bar(x_positions_nb, nb_overfit, width=0.8, label="Naive Bayes", color='skyblue')
    plt.bar(x_positions_svm, svm_overfit, width=0.8, label="SVM", color='coral')
    plt.xlabel("Fold Number")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.title(f"Overfitting Differences per Fold ({n_splits}-Fold)")
    plt.xticks(x_positions_nb + 0.4, [f"Fold {i + 1}" for i in fold_indices])
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    finalize_plot(save_plots, filename="kfold_overfitting_comparison.png" if save_plots else None)

    return df_compare
