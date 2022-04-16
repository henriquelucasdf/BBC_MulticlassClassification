# Custom Transformer for sklearn pipeline
import re
import unicodedata
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom class for text cleaning. 
    
    Parameters:
        data: the list of strings to be cleaned
        normalize: normalize the text (NFKD normalization). Defaults to True
        remove_punct: remove punctuation from the text. Defaults to True
        remove_stopwords: If True, remove stopwords from the text. Defaults to True
        language: the language to use for stopwords. Defaults to 'english'
    
    Attributes:
        cleaned_text: the list containing the cleaned texts after the transformation

        """

    def __init__(self, normalize=True, remove_punct=True, remove_stopwords=True,
        language='english'):

        self.normalize = normalize
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.language = language


    def fit(self, X, y = None):
            return self
    

    def transform(self, X, y=None):
        """
        It takes a list of strings in X, cleans the texts in it and returns the list.

        Args:
            X (list): list of texts to be cleaned
            y (None): not used. Necessary to the sklearn base.  
        
        Returns:
            list: A list with the cleaned texts.
        """
        if not isinstance(X, list):
            raise ValueError("The input must be a List!")

        # copying the input
        X_ = X.copy()
        if self.normalize:
            X_ = list(map(self.__normalize_text, X_))

        if self.remove_punct:
            X_ = list(map(self.__remove_punctuation, X_))
        
        if self.remove_stopwords:
            X_ = [self.__remove_stopwords(text, language=self.language, lower=True) 
                for text in X_]
        
        # attribute cleaned_text
        self.cleaned_text = X_

        return X_

    def __normalize_text(self, text):
        """
        It takes a string and returns a new string that is the same as the original string, except that it
        has been normalized to the NFKD form.
        
        Args:
            text (str): The text to be normalized.
        
        Returns:
            str: The unicode normalization form of the text.
        """
        return unicodedata.normalize('NFKD', text)

    def __remove_punctuation(self, text):
        """
        It removes all punctuation from the text
        
        Args:
            text (str): The text to be processed.
        
        Returns:
            str: the text with all punctuation removed.
        """
        return re.sub(r"[^\w\s]","", text)


    def __remove_stopwords(self, text, language='english', lower=True):
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
        try:
            stop_ = set(stopwords.words(language))
        except Exception:
            # if error, download the language package
            nltk.download(language, quiet=True)
            stop_ = set(stopwords.words(language))

        # removing stopwords
        stop_text = [word for word in tokenized_text if word not in stop_]

        return ' '.join(stop_text)
