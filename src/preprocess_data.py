import re
import unicodedata
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import partial

from sklearn import datasets


def clean_text_for_tfidf(data, normalize=True, remove_punct=True,
    remove_stopwords=True, language='english'):
    """
    It takes a list of strings, clean it and return a list.
    
    Args:
      data: the list of strings to be cleaned
      normalize: normalize the text (NFKD normalization). Defaults to True
      remove_punct: remove punctuation from the text. Defaults to True
      remove_stopwords: If True, remove stopwords from the text. Defaults to True
      language: the language to use for stopwords. Defaults to 'english'
    
    Returns:
      list: A list with the cleaned texts.
    """

    if normalize:
        data = list(map(_normalize_text, data))

    if remove_punct:
        data = list(map(_remove_punctuation, data))
    
    if remove_stopwords:
        data = [_remove_stopwords(text, language=language, lower=True) 
            for text in data]

    return data

def _normalize_text(text):
    """
    It takes a string and returns a new string that is the same as the original string, except that it
    has been normalized to the NFKD form.
    
    Args:
      text (str): The text to be normalized.
    
    Returns:
      str: The unicode normalization form of the text.
    """
    return unicodedata.normalize('NFKD', text)

def _remove_punctuation(text):
    """
    It removes all punctuation from the text
    
    Args:
      text (str): The text to be processed.
    
    Returns:
      str: the text with all punctuation removed.
    """
    return re.sub(r"[^\w\s]","", text)


def _remove_stopwords(text, language='english', lower=True):
    """
    It takes a string, tokenizes it, removes stopwords, and returns a string
    
    Args:
      text (str): the text to be cleaned
      language (str): the language of the text. Defaults to english
      lower (bool): if True, the text will be lowercased. Defaults to True
    
    Returns:
      str: A string of the text with the stopwords removed.
    """
    
    # lower case
    if lower:
        text = text.lower()
    
    # tokenizing (to list)
    tokenized_text = word_tokenize(text, language=language)

    # stopwords (set)
    stop_ = set(stopwords.words(language))

    # removing stopwords
    stop_text = [word for word in tokenized_text if word not in stop_]

    return ' '.join(stop_text)





