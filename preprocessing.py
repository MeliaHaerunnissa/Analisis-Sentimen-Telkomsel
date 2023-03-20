import re
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

list_stopwords = stopwords.words('indonesian')
normalizad_word_dict = {}
factory = StemmerFactory()
stemmer = factory.create_stemmer()
normalizad_word = pd.read_excel('H://myflask/TA-Melia/static/uploads/normalisasi.xlsx')

# Preprocessing
def lower(data):
    return str(data).lower()


def remove_punctuation(data):
    data = re.sub(r"b'\@[\w]*", ' ', data)
    data = re.sub(r"b'[\w]*", ' ', data)
    data = re.sub(r'https\:.*$', " ", data)
    data = re.sub(r'[@][A-Za-z0-9]+', ' ', data)
    data = re.sub(r'[~^0-9]', ' ', data)
    data = re.sub(r'\@\$\w\s*', ' ', data)
    data = re.sub(r'[^\w\s]+', ' ', data)
    data = str(data).lower()
    return data

def tokenize(data):
    data = word_tokenize(data)
    data = ' '.join([char for char in data if char not in string.punctuation])
    return data

def remove_stopwords(data):
    list_stopwords = (["b", "d", "xef", "x", "xa","n", "xe",
            "xf","xb", "xad", "xd","xxxxxxx","xba",
            "xc","k", "xcche", "xd xd xaa xd xd xd xd",
            "m","t","xbb","f","xbf","xbd","xbc","xab","xae"])
    list_stopwords = set(list_stopwords)
    data = ' '.join([word for word in data.split() if word not in list_stopwords])
    return data

def normalized_term(data):
    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]

    data = ' '.join([normalizad_word_dict[term] if term in normalizad_word_dict else term for term in data.split()])
    return data

def stem_text(data):
    data = ' '.join([stemmer.stem(word) for word in data.split()])
    return data

def preprocess_data(data):
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = normalized_term(data)
    data = stem_text(data)
    return 