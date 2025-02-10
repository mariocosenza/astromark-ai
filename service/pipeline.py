import logging
import os
import re
import time
from enum import Enum
from threading import Thread
from typing import Tuple, Optional, Dict, Any

import joblib
import pandas as pd
import psutil
import spacy
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from spellchecker import SpellChecker

# Initialize spell checker for Italian
spell = SpellChecker(language='it')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Define directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
TRAINED_DIR = os.path.join(BASE_DIR, '..', 'data', 'trained')

# File paths
TICKET_O3_PATH = os.path.join(DATA_RAW_DIR, 'ticket-o3.csv')
TICKET_GEMINI_PATH = os.path.join(DATA_RAW_DIR, 'ticket-gemini-claude.csv')
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, 'X_processed.csv')

# Ensure necessary directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)

# Load raw datasets
logger.info("Loading ticket-o3 from: %s", TICKET_O3_PATH)
dataframe_o3: pd.DataFrame = pd.read_csv(TICKET_O3_PATH, usecols=['titolo', 'messaggio', 'categoria'])
logger.info("Loaded ticket-o3.csv with shape %s", dataframe_o3.shape)

logger.info("Loading ticket-gemini from: %s", TICKET_GEMINI_PATH)
dataframe_gc: pd.DataFrame = pd.read_csv(TICKET_GEMINI_PATH, usecols=['titolo', 'messaggio', 'categoria'])
logger.info("Loaded ticket-gemini-claude.csv with shape %s", dataframe_gc.shape)


def merge_dataframes(frame1: pd.DataFrame, frame2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two dataframes and combine 'titolo' and 'messaggio' into a single column.
    Returns a dataframe with columns 'titolo_messaggio' and 'categoria'.
    """
    logger.info("Merging dataframes...")
    frame = pd.concat([frame1, frame2])
    num_duplicated = frame.duplicated().sum()
    frame.drop_duplicates(inplace=True)
    logger.info("Eliminated %s duplicate rows", num_duplicated)
    frame['titolo_messaggio'] = frame['titolo'] + ' ' + frame['messaggio']
    return frame[['titolo_messaggio', 'categoria']]


merged_df: pd.DataFrame = merge_dataframes(dataframe_o3, dataframe_gc)
X: pd.Series = merged_df['titolo_messaggio']
y: pd.Series = merged_df['categoria']
logger.info("Final merged dataset shape: %s", merged_df.shape)

###############################################################################
#                         Minimal Text Preprocessing
###############################################################################
logger.info("Loading spaCy model (it_core_news_sm)...")
nlp = spacy.load('it_core_news_sm')  # Includes NER by default

# Patterns to remove common Italian greetings
GREETINGS_SECRETARY_PATTERNS = [
    r'\bciao\b', r'\bbuongiorno\b', r'\bsalve\b',
    r'\bbuonasera\b', r'\bbuon pomeriggio\b', r'\barrivederci\b',
    r'\bbuonanotte\b', r'\ba presto\b', r'\baddio\b', r'\bsaluti\b',
    r'\bspettabile\b', r'\bcordiali saluti\b', r'distinti saluti\b',
    r'\bsalute\b', r'\bsegreteria\b'
]


def remove_greetings_secretary(text: str) -> str:
    """
    Remove common greetings from the text.
    """
    pattern = re.compile('|'.join(GREETINGS_SECRETARY_PATTERNS), flags=re.IGNORECASE)
    return pattern.sub('', text)


def minimal_preprocess(text: str) -> str:
    """
    Perform minimal normalization on the text:
    lowercase, remove URLs, greetings, punctuation, numbers, and extra whitespace,
    then perform a fast spell check for Italian.
    """
    text = text.lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = remove_greetings_secretary(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()

    corrected_tokens = [spell.correction(word) if word in spell else word for word in tokens]
    # to add full spell check use the following lines
    # corrected_tokens = []
    # for word in tokens:
    #    corrected_word = spell.correction(word)
    #    corrected_tokens.append(corrected_word if corrected_word else word)

    return ' '.join(corrected_tokens).strip()


def process_text(text: str) -> str:
    """
    Process text using minimal preprocessing and spaCy for tokenization/lemmatization and NER.
    Appends NER tokens (e.g., "NER_LABEL") to the token list.
    """
    cleaned_text = minimal_preprocess(text)
    doc = nlp(cleaned_text)
    tokens = [token.lemma_.strip() for token in doc if
              not (token.is_stop or token.is_punct or token.is_space) and token.lemma_]
    for ent in doc.ents:
        tokens.append("NER_%s" % ent.label_)
    return ' '.join(tokens)


def parallel_process_texts(series: pd.Series, n_jobs: int = -1) -> pd.Series:
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
    df_cached = pd.read_csv(PROCESSED_DATA_PATH)
    X_processed: pd.Series = df_cached["processed_text"]
else:
    logger.info("Preprocessed data not found, starting parallel preprocessing...")
    X_processed = parallel_process_texts(X, n_jobs=-1)
    logger.info("Saving preprocessed data to: %s", PROCESSED_DATA_PATH)
    pd.DataFrame(X_processed, columns=["processed_text"]).to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Preprocessing complete and cached.")


###############################################################################
#                           Classifier Type Enum
###############################################################################
class ClassifierType(Enum):
    """Enumeration of classifier types."""
    NAIVE_BAYES = "naive_bayes"
    SVM = "svm"


###############################################################################
#                Model Pipeline + Grid Search (Training)
###############################################################################
def build_pipeline(classifier_type: ClassifierType) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Build a machine learning pipeline for the specified classifier type.
    Returns a tuple of (pipeline, parameter grid) for grid search.
    """
    tfidf = TfidfVectorizer(
        use_idf=True,
        ngram_range=(1, 2),
        max_features=3000,
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
        svd = TruncatedSVD(n_components=100, random_state=42)
        classifier = SVC(probability=True, kernel='linear', random_state=42)
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('svd', svd),
            ('clf', classifier)
        ])
        param_grid = {
            'tfidf__min_df': [1, 3, 10],
            'tfidf__max_df': [0.85, 0.90],
            'svd__n_components': [30, 50, 100, 150],
            'clf__C': [0.1, 0.5, 1.0, 2.0]
        }
    else:
        raise ValueError("Unsupported classifier type.")

    return pipeline, param_grid


def _run_grid_search(x_data: pd.Series,
                     y_data: pd.Series,
                     classifier_type: ClassifierType,
                     monitor: bool = False
                     ) -> Tuple[GridSearchCV, Optional[Dict[str, Any]]]:
    """
    Run grid search for the given classifier type.
    If monitor is True, concurrently measure CPU (per logical core) and memory usage.
    Returns a tuple of (grid_search_object, monitor_data).
    When monitor is False, monitor_data is None.
    """
    pipeline, param_grid = build_pipeline(classifier_type)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=kf, n_jobs=-1, verbose=0)
    monitor_data: Optional[Dict[str, Any]] = None

    if monitor:
        process = psutil.Process(os.getpid())
        num_cpus = psutil.cpu_count(logical=True)
        cpu_usage = [[] for _ in range(num_cpus)]
        mem_usage = []  # in MB
        time_list = []
        keep_monitoring = True

        def monitor_usage() -> None:
            while keep_monitoring:
                usage = psutil.cpu_percent(interval=None, percpu=True)
                for i, u in enumerate(usage):
                    cpu_usage[i].append(u)
                mem_usage.append(process.memory_info().rss / (1024 * 1024))
                time_list.append(time.time())
                time.sleep(0.2)

        monitor_thread = Thread(target=monitor_usage)
        monitor_thread.start()
        start_time = time.time()
        grid.fit(x_data, y_data)
        end_time = time.time()
        keep_monitoring = False
        monitor_thread.join()
        monitor_data = {
            "cpu_usage": cpu_usage,
            "mem_usage": mem_usage,
            "time_list": time_list,
            "elapsed": end_time - start_time,
        }
    else:
        grid.fit(x_data, y_data)

    return grid, monitor_data


def perform_grid_search(x_data: pd.Series,
                        y_data: pd.Series,
                        classifier_type: ClassifierType) -> Pipeline:
    """
    Perform grid search with cross-validation for the given classifier type.
    Returns the best estimator.
    """
    logger.info("Building pipeline for %s...", classifier_type.value)
    grid, _ = _run_grid_search(x_data, y_data, classifier_type, monitor=False)
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


def save_model(model: Pipeline, classifier_type: ClassifierType) -> None:
    """
    Save the trained model to disk.
    """
    path = MODEL_PATHS[classifier_type.value]
    joblib.dump(model, path)
    logger.info("Model saved to %s.", path)


def load_model(classifier_type: ClassifierType) -> Optional[Pipeline]:
    """
    Load a previously saved model from disk, if it exists.
    """
    path = MODEL_PATHS[classifier_type.value]
    if os.path.exists(path):
        logger.info("Loading saved model from %s...", path)
        return joblib.load(path)
    return None


def get_model(classifier_type: ClassifierType) -> Pipeline:
    """
    Load a pre-trained model if available; otherwise, train via grid search and save the model.
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


###############################################################################
#                      Measure PC Metrics During Training
###############################################################################
def measure_pc_metrics_during_training(classifier_type: ClassifierType,
                                       save_plot: bool = False) -> Pipeline:
    """
    Train the specified classifier type while measuring per-core CPU and RAM usage.
    Optionally save the usage plots.
    Returns the trained (best) model.
    """
    logger.info("Measuring CPU & RAM usage while training %s...", classifier_type.value)
    grid, monitor_data = _run_grid_search(X_processed, y, classifier_type, monitor=True)
    best_model = grid.best_estimator_
    logger.info("CPU & RAM usage measurement complete.")
    logger.info("Training finished in %.2f seconds.", monitor_data["elapsed"])
    logger.info("Best parameters for %s: %s", classifier_type.value, grid.best_params_)

    # Normalize time: convert to elapsed seconds from start
    t0 = monitor_data["time_list"][0]
    times_elapsed = [t - t0 for t in monitor_data["time_list"]]

    import matplotlib.pyplot as plt

    # Plot CPU usage per logical core
    num_cpus = len(monitor_data["cpu_usage"])
    plt.figure(figsize=(10, 6))
    for i in range(num_cpus):
        plt.plot(times_elapsed, monitor_data["cpu_usage"][i], label=f"CPU {i}")
    plt.xlabel("Seconds (relative to start)")
    plt.ylabel("CPU Usage (%)")
    plt.title(f"CPU Usage During Training ({classifier_type.value})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_plot:
        cpu_plot_filename = f"cpu_usage_{classifier_type.value}.png"
        plt.savefig(cpu_plot_filename)
        plt.close()
        logger.info("CPU usage plot saved to '%s'", cpu_plot_filename)
    else:
        plt.show()

    # Plot Memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(times_elapsed, monitor_data["mem_usage"], color='blue', label="Memory (MB)")
    plt.xlabel("Seconds (relative to start)")
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"Memory Usage During Training ({classifier_type.value})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_plot:
        mem_plot_filename = f"mem_usage_{classifier_type.value}.png"
        plt.savefig(mem_plot_filename)
        plt.close()
        logger.info("Memory usage plot saved to '%s'", mem_plot_filename)
    else:
        plt.show()

    save_model(best_model, classifier_type)
    return best_model


def select_default_classifier() -> Pipeline:
    # Select classifier and retrieve final model (load if it exists)
    selected_classifier: ClassifierType = ClassifierType.SVM
    logger.info("Selected classifier: %s", selected_classifier.value)
    logger.info("Retrieving final model (will load if it already exists)...")
    return get_model(selected_classifier)
