import logging
import os
import re
from enum import Enum

import joblib
import pandas as pd
import spacy
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from spellchecker import SpellChecker

# Initialize the spell checker for Italian
spell = SpellChecker(language='it')

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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

logger.info("Loading ticket-o3 from: %s", TICKET_O3_PATH)
dataframe_o3 = pd.read_csv(TICKET_O3_PATH, usecols=['titolo', 'messaggio', 'categoria'])
logger.info("Loaded ticket-o3.csv with shape %s", dataframe_o3.shape)

logger.info("Loading ticket-gemini from: %s", TICKET_GEMINI_PATH)
dataframe_gc = pd.read_csv(TICKET_GEMINI_PATH, usecols=['titolo', 'messaggio', 'categoria'])
logger.info("Loaded ticket-gemini-claude.csv with shape %s", dataframe_gc.shape)


def merge_dataframes(frame1, frame2):
    """
    Merge two dataframes and combine 'titolo' and 'messaggio' into a single column.

    Returns a dataframe with columns 'titolo_messaggio' and 'categoria'.
    """
    logger.info("Merging dataframes...")
    frame = pd.concat([frame1, frame2])

    num_duplicated = frame.duplicated().sum()
    frame.drop_duplicates(inplace=True)
    logger.info("Eliminate", str(num_duplicated),"righe duplicate")

    frame['titolo_messaggio'] = frame['titolo'] + ' ' + frame['messaggio']
    return frame[['titolo_messaggio', 'categoria']]


merged_df = merge_dataframes(dataframe_o3, dataframe_gc)
X = merged_df['titolo_messaggio']
y = merged_df['categoria']
logger.info("Final merged dataset shape: %s", merged_df.shape)

###############################################################################
#                         Minimal Text Preprocessing
###############################################################################
logger.info("Loading spaCy model (it_core_news_sm)...")
nlp = spacy.load('it_core_news_sm')  # Includes NER by default

# Common Italian greetings to remove
GREETINGS_SECRETARY_PATTERNS = [
    r'\bciao\b', r'\bbuongiorno\b', r'\bsalve\b',
    r'\bbuonasera\b', r'\bbuon pomeriggio\b', r'\barrivederci\b',
    r'\bbuonanotte\b', r'\ba presto\b', r'\baddio\b', r'\bsaluti\b',
    r'\bspettabile\b', r'\bcordiali saluti\b', r'distinti saluti\b', r'\bsalve\b', r'\bsegreteria\b'
]


def remove_greetings_secretary(text):
    """
    Remove common greetings from the text.
    """
    pattern = re.compile('|'.join(GREETINGS_SECRETARY_PATTERNS), flags=re.IGNORECASE)
    return pattern.sub('', text)





def minimal_preprocess(text):
    """
    Perform minimal normalization on the text:
    lowercase, remove URLs, greetings, punctuation, numbers, and extra whitespace,
    followed by a fast spell check for Italian text.
    """
    # Step 1: Normalize text
    text = text.lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = remove_greetings_secretary(text)  # Remove greetings
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces

    # Step 2: Tokenize and perform spell check
    tokens = text.split()  # Split text into words
    corrected_tokens = [spell.correction(word) if word in spell else word for word in tokens]

    # Step 3: Reconstruct the corrected text
    corrected_text = ' '.join(corrected_tokens)

    return corrected_text.strip()


def process_text(text):
    """
    Process text using minimal preprocessing and spaCy for tokenization/lemmatization and NER.
    Appends NER tokens in the form "NER_LABEL" to the token list.
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
    for ent in doc.ents:
        tokens.append("NER_%s" % ent.label_)
    return ' '.join(tokens)


def parallel_process_texts(series, n_jobs=-1):
    """
    Apply process_text to each element in a pandas Series in parallel using threading.

    Returns a pandas Series of processed texts.
    """
    logger.info("Parallel text processing with threading backend...")
    with parallel_backend('threading', n_jobs=n_jobs):
        processed = Parallel()(delayed(process_text)(text) for text in series)
    return pd.Series(processed, index=series.index)


###############################################################################
#                        Load / Save Processed Dataset
###############################################################################
if os.path.exists(PROCESSED_DATA_PATH):
    logger.info("Loading preprocessed data from '%s'...", PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X_processed = df["processed_text"]
else:
    logger.info("Preprocessed data not found, starting parallel preprocessing...")
    X_processed = parallel_process_texts(X, n_jobs=-1)
    logger.info("Saving preprocessed data to: %s", PROCESSED_DATA_PATH)
    X_processed_df = pd.DataFrame(X_processed, columns=["processed_text"])
    X_processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Preprocessing complete and cached.")


###############################################################################
#                           Classifier Type Enum
###############################################################################
class ClassifierType(Enum):
    """
    Enumeration of classifier types.
    """
    NAIVE_BAYES = "naive_bayes"
    SVM = "svm"


###############################################################################
#                Model Pipeline + Grid Search (Training)
###############################################################################
def build_pipeline(classifier_type):
    """
    Build a machine learning pipeline for the specified classifier type.

    Returns a tuple of (pipeline, parameter grid) for grid search.
    """
    tfidf = TfidfVectorizer(
        use_idf=True,
        ngram_range=(1, 2),
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


def perform_grid_search(x_data, y_data, classifier_type):
    """
    Perform grid search with cross-validation for the given classifier type.

    Returns the best estimator from the grid search.
    """
    logger.info("Building pipeline for %s...", classifier_type.value)
    pipeline, param_grid = build_pipeline(classifier_type)
    logger.info("Starting grid search for %s with parameters: %s", classifier_type.value, param_grid)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=-1, verbose=1)
    grid.fit(x_data, y_data)
    logger.info("Grid search for %s complete.", classifier_type.value)
    logger.info("Best parameters: %s", grid.best_params_)
    return grid.best_estimator_


###############################################################################
#                          Save/Load Final Model
###############################################################################
MODEL_PATHS = {
    "naive_bayes": os.path.join(TRAINED_DIR, "trained_model_nb.pkl"),
    "svm": os.path.join(TRAINED_DIR, "trained_model_svm.pkl")
}


def save_model(model, classifier_type):
    """
    Save the trained model to disk.
    """
    path = MODEL_PATHS[classifier_type.value]
    joblib.dump(model, path)
    logger.info("Model saved to %s.", path)


def load_model(classifier_type):
    """
    Load a previously saved model from disk, if it exists.
    """
    path = MODEL_PATHS[classifier_type.value]
    if os.path.exists(path):
        logger.info("Loading saved model from %s...", path)
        return joblib.load(path)
    return None


def get_model(classifier_type):
    """
    Load a pre-trained model if available; otherwise, perform grid search and save the model.

    Returns the model.
    """
    model = load_model(classifier_type)
    if model is None:
        logger.info("No saved model found for %s. Training a new one...", classifier_type.value)
        model = perform_grid_search(X_processed, y, classifier_type)
        save_model(model, classifier_type)
    else:
        logger.info("Using saved model for %s.", classifier_type.value)
    return model


# Select the default classifier (SVM in this case)
selected_classifier = ClassifierType.SVM
logger.info("Selected classifier: %s", selected_classifier.value)

logger.info("Retrieving final model (will load if already exists)...")
final_model = get_model(selected_classifier)
