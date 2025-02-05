# report_and_predict.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, accuracy_score
)

# Import the relevant objects from pipeline.py
from .pipeline import (
    get_model,
    process_text,
    X_processed,
    y,
    final_model,
    ClassifierType
)


def generate_full_model_report(model, X_train, y_train, X_test, y_test, save_plots=False):
    """
    Generates and prints a comprehensive report of model performance on both training and test sets.
    Includes:
      - Classification Report (as a DataFrame)
      - Accuracy
      - Confusion Matrix (printed and plotted as a heatmap)
      - ROC Curve and AUC (for binary classification)
      - A bar plot for per-class precision, recall, and f1-score (from the test set)

    Plots are only shown; they are not saved to disk.

    Returns a dictionary with detailed metrics for further analysis.
    """
    print("[INFO] Generating full model report...")

    # ---------------------------
    # Predict on train set
    # ---------------------------
    y_train_pred = model.predict(X_train)
    train_report_dict = classification_report(y_train, y_train_pred, output_dict=True)
    train_report_df = pd.DataFrame(train_report_dict).transpose()
    train_conf = confusion_matrix(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Check if the classifier is in a pipeline under 'clf'
    try:
        clf = model.named_steps['clf']
    except Exception:
        clf = model

    if hasattr(clf, 'classes_') and len(clf.classes_) == 2:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        auc_train = auc(fpr_train, tpr_train)
    else:
        auc_train = None

    # ---------------------------
    # Predict on test set
    # ---------------------------
    y_test_pred = model.predict(X_test)
    test_report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    test_report_df = pd.DataFrame(test_report_dict).transpose()
    test_conf = confusion_matrix(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    if hasattr(clf, 'classes_') and len(clf.classes_) == 2:
        y_test_prob = model.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
        auc_test = auc(fpr_test, tpr_test)
    else:
        auc_test = None

    # ---------------------------
    # Print Metrics
    # ---------------------------
    print("\n========== TRAINING METRICS ==========")
    print("Overall Accuracy: {:.4f}".format(train_accuracy))
    if auc_train is not None:
        print("ROC AUC: {:.4f}".format(auc_train))
    print("\nClassification Report (Train):")
    print(train_report_df)
    if hasattr(clf, 'classes_'):
        print("\nConfusion Matrix (Train):\nClasses:", clf.classes_)
    print(train_conf)

    print("\n========== TESTING METRICS ==========")
    print("Overall Accuracy: {:.4f}".format(test_accuracy))
    if auc_test is not None:
        print("ROC AUC: {:.4f}".format(auc_test))
    print("\nClassification Report (Test):")
    print(test_report_df)
    if hasattr(clf, 'classes_'):
        print("\nConfusion Matrix (Test):\nClasses:", clf.classes_)
    print(test_conf)

    # ---------------------------
    # Plot Confusion Matrices
    # ---------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(train_conf, annot=True, fmt="d", cmap="Blues")
    plt.title("Training Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(test_conf, annot=True, fmt="d", cmap="Blues")
    plt.title("Testing Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ---------------------------
    # Plot ROC Curves (for binary classification)
    # ---------------------------
    if auc_train is not None and auc_test is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, label="Train ROC (AUC = {:.4f})".format(auc_train))
        plt.plot(fpr_test, tpr_test, label="Test ROC (AUC = {:.4f})".format(auc_test))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    # ---------------------------
    # Additional Plot: Bar Chart for per-class metrics (precision, recall, f1)
    # ---------------------------
    # Exclude aggregate rows from the classification report
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
        plt.show()

    return {
        "train": {
            "accuracy": train_accuracy,
            "auc": auc_train,
            "report_df": train_report_df,
            "confusion_matrix": train_conf
        },
        "test": {
            "accuracy": test_accuracy,
            "auc": auc_test,
            "report_df": test_report_df,
            "confusion_matrix": test_conf
        }
    }


def predict_category(message, classifier_type, top_n=3):
    """
    Preprocesses the input message and predicts its category using the specified classifier.
    If the model supports predict_proba and has multiple classes, returns the top 'top_n'
    predicted categories with their probabilities. Otherwise, returns a single label.
    """
    print("[INFO] Predicting category for a new message...")
    model = get_model(classifier_type)
    processed_message = process_text(message)  # minimal_preprocess + spaCy + NER

    # Check if the classifier supports probabilities
    try:
        clf = model.named_steps['clf']
    except Exception:
        clf = model

    if hasattr(clf, "predict_proba"):
        probs = model.predict_proba([processed_message])[0]
        classes = clf.classes_
        # Sort by probability (descending)
        sorted_indices = np.argsort(probs)[::-1]
        top_n = min(top_n, len(classes))
        top_indices = sorted_indices[:top_n]

        # Build a list of (class, probability)
        predictions = [(classes[i], probs[i]) for i in top_indices]
        return predictions
    else:
        category = model.predict([processed_message])[0]
        return [(category, 1.0)]  # 100% for that predicted class


# usage_example.py (hypothetical script in the same folder)


from sklearn.model_selection import train_test_split

# Split for evaluation
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Report
report = generate_full_model_report(final_model, X_train_eval, y_train_eval, X_test_eval, y_test_eval)

# Predict
test_message = "Mio figlio non riesce a vedere il voto di Matematica"
predictions = predict_category(test_message, ClassifierType.SVM)
