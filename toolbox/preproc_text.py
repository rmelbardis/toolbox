'''
Functions used to preprocess text for machine learning.
'''

import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download

def remove_punc(text):
    txt = text
    for punc in string.punctuation:
        txt = txt.replace(punc, '')
    return txt

def remove_num(text):
    return ''.join(char for char in text if not char.isdigit())

def remove_stopw(text):
    download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    return ' '.join(w for w in word_tokens if not w in stop_words)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))

def preprocess_text(series):
    step = series.apply(remove_punc).apply(remove_num).str.lower()
    return step.apply(remove_stopw).apply(lemmatize)
