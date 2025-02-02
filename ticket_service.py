import pandas as pd
import numpy as np
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

dataframe_o3 = pd.read_csv("data/raw/ticket-o3.csv", usecols = ['titolo','messaggio','categoria'])
dataframe_gc = pd.read_csv("data/raw/ticket-gemini-claude.csv", usecols = ['titolo','messaggio','categoria'])

def merge_dataframes(frame1, frame2):
    return pd.concat([frame1, frame2])


def split_dataframe(dataframe):
    train_set, test_set = train_test_split(dataframe, random_state=42, test_size=0.2)
    print(train_set.shape, test_set.shape)
