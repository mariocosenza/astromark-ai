import os
import re
import joblib
import pandas as pd
import spacy
from enum import Enum
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Get the directory of this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Construct paths relative to this file
DATA_RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
TRAINED_DIR = os.path.join(BASE_DIR, '..', 'data', 'trained')

TICKET_O3_PATH = os.path.join(DATA_RAW_DIR, 'ticket-o3.csv')
TICKET_GEMINI_PATH = os.path.join(DATA_RAW_DIR, 'ticket-gemini-claude.csv')
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, 'X_processed.csv')

# Ensure necessary directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)

print(f"[INFO] Loading ticket-o3 from: {TICKET_O3_PATH}")
dataframe_o3 = pd.read_csv(TICKET_O3_PATH, usecols=['titolo', 'messaggio', 'categoria'])
print(f"[INFO] Loaded ticket-o3.csv with shape {dataframe_o3.shape}")

print(f"[INFO] Loading ticket-gemini from: {TICKET_GEMINI_PATH}")
dataframe_gc = pd.read_csv(TICKET_GEMINI_PATH, usecols=['titolo', 'messaggio', 'categoria'])
print(f"[INFO] Loaded ticket-gemini-claude.csv with shape {dataframe_gc.shape}")

def merge_dataframes(frame1, frame2):
    print("[INFO] Merging dataframes...")
    frame = pd.concat([frame1, frame2])
    frame['titolo_messaggio'] = frame['titolo'] + ' ' + frame['messaggio']
    return frame[['titolo_messaggio', 'categoria']]

merged_df = merge_dataframes(dataframe_o3, dataframe_gc)
X = merged_df['titolo_messaggio']
y = merged_df['categoria']
print(f"[INFO] Final merged dataset shape: {merged_df.shape}")

###############################################################################
#                         Minimal Text Preprocessing
###############################################################################
print("[INFO] Loading spaCy model (it_core_news_sm)...")
nlp = spacy.load('it_core_news_sm')  # Includes NER by default

# Common Italian greetings to remove
GREETINGS_PATTERNS = [
    r'\bciao\b', r'\bbuongiorno\b', r'\bsalve\b',
    r'\bbuonasera\b', r'\bbuon pomeriggio\b', r'\barrivederci\b',
    r'\bbuonanotte\b', r'\ba presto\b', r'\baddio\b', r'\bsaluti\b'
]

def remove_greetings(text):
    pattern = re.compile('|'.join(GREETINGS_PATTERNS), flags=re.IGNORECASE)
    return pattern.sub('', text)

def minimal_preprocess(text):
    """
    Minimal normalization: lowercase, remove URLs, greetings,
    punctuation, numbers, and extra whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = remove_greetings(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_text(text):
    """
    Processes the text using minimal_preprocess and spaCy for tokenization/lemmatization and NER.
    NER tokens are appended in the form "NER_LABEL".
    """
    cleaned_text = minimal_preprocess(text)
    doc = nlp(cleaned_text)

    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_
        if lemma:
            lemma = lemma.strip()
            if lemma:
                tokens.append(lemma)
    # Append NER tokens
    for ent in doc.ents:
        tokens.append(f"NER_{ent.label_}")
    return ' '.join(tokens)

def parallel_process_texts(series, n_jobs=-1):
    """
    Applies process_text to each element in a pandas Series in parallel using threads.
    """
    print("[INFO] Parallel text processing with threading backend...")
    with parallel_backend('threading', n_jobs=n_jobs):
        processed = Parallel()(delayed(process_text)(text) for text in series)
    return pd.Series(processed, index=series.index)

###############################################################################
#                        Load / Save Processed Dataset
###############################################################################
if os.path.exists(PROCESSED_DATA_PATH):
    print(f"[INFO] Loading preprocessed data from '{PROCESSED_DATA_PATH}'...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X_processed = df["processed_text"]
else:
    print("[INFO] Preprocessed data not found, starting parallel preprocessing...")
    X_processed = parallel_process_texts(X, n_jobs=-1)
    print("[INFO] Saving preprocessed data to:", PROCESSED_DATA_PATH)
    X_processed_df = pd.DataFrame(X_processed, columns=["processed_text"])
    X_processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("[INFO] Preprocessing complete and cached.")

###############################################################################
#                           Classifier Type Enum
###############################################################################
class ClassifierType(Enum):
    NAIVE_BAYES = "naive_bayes"
    SVM = "svm"

###############################################################################
#                Model Pipeline + Grid Search (Training)
###############################################################################
def build_pipeline(classifier_type):
    """
    Builds a pipeline with TF-IDF vectorization and (optionally) dimensionality reduction (for SVM)
    along with the specified classifier.
    """
    tfidf = TfidfVectorizer(
        use_idf=True,
        ngram_range=(1, 1),
        max_features=2000,
        norm='l2',
        smooth_idf=True,
        sublinear_tf=True
    )

    if classifier_type == ClassifierType.NAIVE_BAYES:
        classifier = MultinomialNB()
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', classifier)
        ])
        param_grid = {
            'tfidf__min_df': [1, 3],
            'tfidf__max_df': [0.85, 0.90],
            'clf__alpha': [1.0, 1.5, 2.0]
        }
    elif classifier_type == ClassifierType.SVM:
        svd = TruncatedSVD(n_components=50, random_state=42)
        classifier = SVC(probability=True, kernel='linear', random_state=42)
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('svd', svd),
            ('clf', classifier)
        ])
        param_grid = {
            'tfidf__min_df': [1, 3],
            'tfidf__max_df': [0.85, 0.90],
            'svd__n_components': [30, 50],
            'clf__C': [0.1, 0.5, 1.0]
        }
    else:
        raise ValueError("Unsupported classifier type.")
    return pipeline, param_grid

def perform_grid_search(X, y, classifier_type):
    print(f"[INFO] Building pipeline for {classifier_type.value}...")
    pipeline, param_grid = build_pipeline(classifier_type)
    print(f"[INFO] Starting grid search for {classifier_type.value} with parameters: {param_grid}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(f"[INFO] Grid search for {classifier_type.value} complete.")
    print(f"[INFO] Best parameters: {grid.best_params_}")
    return grid.best_estimator_

###############################################################################
#                          Save/Load Final Model
###############################################################################
MODEL_PATHS = {
    ClassifierType.NAIVE_BAYES: os.path.join(TRAINED_DIR, "trained_model_nb.pkl"),
    ClassifierType.SVM: os.path.join(TRAINED_DIR, "trained_model_svm.pkl")
}

def save_model(model, classifier_type):
    path = MODEL_PATHS[classifier_type]
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}.")

def load_model(classifier_type):
    path = MODEL_PATHS[classifier_type]
    if os.path.exists(path):
        print(f"[INFO] Loading saved model from {path}...")
        return joblib.load(path)
    return None

def get_model(classifier_type):
    """
    Loads a pre-trained model if available; otherwise, performs grid search and saves the model.
    """
    model = load_model(classifier_type)
    if model is None:
        print(f"[INFO] No saved model found for {classifier_type.value}. Training a new one...")
        model = perform_grid_search(X_processed, y, classifier_type)
        save_model(model, classifier_type)
    else:
        print(f"[INFO] Using saved model for {classifier_type.value}.")
    return model

# By default, let's choose SVM
selected_classifier = ClassifierType.SVM
print(f"[INFO] Selected classifier: {selected_classifier.value}")

print("[INFO] Retrieving final model (will load if already exists)...")
final_model = get_model(selected_classifier)
