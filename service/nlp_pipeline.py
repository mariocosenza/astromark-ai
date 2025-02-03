import os
import joblib
import pandas as pd
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
dataframe_o3 = pd.read_csv("../data/raw/ticket-o3.csv", usecols=['titolo', 'messaggio', 'categoria'])
dataframe_gc = pd.read_csv("../data/raw/ticket-gemini-claude.csv", usecols=['titolo', 'messaggio', 'categoria'])


def merge_dataframes(frame1, frame2):
    frame = pd.concat([frame1, frame2])
    frame['titolo_messaggio'] = frame['titolo'] + ' ' + frame['messaggio']
    frame = frame[['titolo_messaggio', 'categoria']]
    return frame


# Merge the dataframes
merged_df = merge_dataframes(dataframe_o3, dataframe_gc)

# Split the data
train_df, test_df = train_test_split(merged_df, random_state=42, test_size=0.2)
print("Train and test sizes:", train_df.shape, test_df.shape)

# Separate features and target
X_train = train_df['titolo_messaggio']
y_train = train_df['categoria']
X_test = test_df['titolo_messaggio']
y_test = test_df['categoria']

# Load SpaCy Italian model
nlp = spacy.load('it_core_news_sm')


def preprocess(text):
    """Basic text cleaning: convert to lowercase, remove punctuation, and extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_text(text):
    """Process text using spaCy: cleaning, lemmatizing, and removing stop words and punctuation."""
    text = preprocess(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)


def get_tfidf_vectors(X_train, X_test):
    """Generate TF-IDF vectors for train and test datasets."""
    tfidf_vectorizer = TfidfVectorizer(
        use_idf=True,
        ngram_range=(1, 2)
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer


def predict_category(message):
    """
    Given an input message string, preprocess it and predict the category.

    Parameters:
       message (str): The input message to be classified.

    Returns:
       category: The predicted category from the trained model.
    """
    # Process the message using the shared pipeline.
    processed_message = process_text(message)

    # Transform the processed message to TF-IDF vector format.
    message_vector = tfidf_vectorizer.transform([processed_message])

    # Predict the category using the trained model.
    category = nb_tfidf.predict(message_vector)[0]
    return category

# Process text data
X_train_processed = X_train.apply(process_text)
X_test_processed = X_test.apply(process_text)

# Define paths to save the model and vectorizer
model_path = "nb_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    # Load the saved model and vectorizer if they exist.
    nb_tfidf = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    X_train_vectors_tfidf = tfidf_vectorizer.transform(X_train_processed)
    X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test_processed)
    print("Loaded model and vectorizer from disk.")
else:
    # Generate TF-IDF vectors and train the model.
    X_train_vectors_tfidf, X_test_vectors_tfidf, tfidf_vectorizer = get_tfidf_vectors(
        X_train_processed, X_test_processed
    )
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_vectors_tfidf, y_train)

    # Save the model and vectorizer for future use.
    joblib.dump(nb_tfidf, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print("Model and vectorizer have been trained and saved.")

# Predict on test data
y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)

# If binary classification, calculate AUC (assuming positive class is at index 1)
if len(nb_tfidf.classes_) == 2:
    y_prob = y_prob[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

# Print classification metrics
print("\nClassification Report:")
print(classification_report(y_test, y_predict))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_predict))
print(predict_category("Buongiorno, mio figlio non riesce ad accedere al suo profilo"))