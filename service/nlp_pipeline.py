import os
import re
import numpy as np
import joblib
import pandas as pd
import spacy
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB

# Load data
dataframe_o3 = pd.read_csv("../data/raw/ticket-o3.csv", usecols=['titolo', 'messaggio', 'categoria'])
dataframe_gc = pd.read_csv("../data/raw/ticket-gemini-claude.csv", usecols=['titolo', 'messaggio', 'categoria'])

# Merge dataframes
def merge_dataframes(frame1, frame2):
    frame = pd.concat([frame1, frame2])
    frame['titolo_messaggio'] = frame['titolo'] + ' ' + frame['messaggio']
    return frame[['titolo_messaggio', 'categoria']]

merged_df = merge_dataframes(dataframe_o3, dataframe_gc)

# Separate features and target
X = merged_df['titolo_messaggio']
y = merged_df['categoria']

# Load SpaCy model
nlp = spacy.load('it_core_news_sm')

# Text preprocessing
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_text(text):
    text = preprocess(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

# K-Fold settings
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Store evaluation metrics
all_reports = []
all_confusion_matrices = []
auc_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Processing Fold {fold + 1}...")

    # Split data into train and test for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Process text
    X_train_processed = X_train.apply(process_text)
    X_test_processed = X_test.apply(process_text)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_processed)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_processed)

    # Train model
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = nb_tfidf.predict(X_test_tfidf)
    y_prob = nb_tfidf.predict_proba(X_test_tfidf)

    # Compute AUC if binary classification
    if len(nb_tfidf.classes_) == 2:
        y_prob = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)

    # Convert classification report into DataFrame and store it
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    all_reports.append(report_df)

    # Store confusion matrix
    all_confusion_matrices.append(confusion_matrix(y_test, y_pred))

# Compute average metrics
avg_report_df = pd.concat(all_reports).groupby(level=0).mean()

# Compute average confusion matrix
avg_conf_matrix = np.mean(all_confusion_matrices, axis=0)

# Print summary
print("\n===== K-Fold Cross Validation Results =====")
if auc_scores:
    print(f"Average AUC: {np.mean(auc_scores):.4f}")

print("\nAverage Classification Report:")
print(avg_report_df)

print("\nAverage Confusion Matrix:")
print(avg_conf_matrix)

def predict_category(message, model, vectorizer):
    """
    Dato un messaggio in input, lo pre-processa e ne predice la categoria.

    Parameters:
       message (str): Il messaggio da classificare.
       model (MultinomialNB): Il modello Naive Bayes allenato.
       vectorizer (TfidfVectorizer): Il vettorizzatore TF-IDF addestrato.

    Returns:
       str: La categoria predetta.
    """
    # Processa il testo come nel training
    processed_message = process_text(message)

    # Trasforma il testo in vettore TF-IDF
    message_vector = vectorizer.transform([processed_message])

    # Predice la categoria
    category = model.predict(message_vector)[0]

    return category

test_message = "Ho un problema con il login, non riesco ad accedere."
predicted_category = predict_category(test_message, nb_tfidf, tfidf_vectorizer)
print("La categoria predetta è:", predicted_category)
