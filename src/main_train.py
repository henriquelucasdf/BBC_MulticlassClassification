import os
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
import preprocess_data

def get_script_arguments():

    # Initializing
    parser = argparse.ArgumentParser()

    # adding arguments
    parser.add_argument('--data_relative_path', type=str, default='data/')
    parser.add_argument('--random_state', type=int, default=1)

    # parsing
    script_args = parser.parse_args()

    return script_args

def get_script_path():
    """
    Get the absolute path of the script directory
    
    Returns:
      The absolute path of the directory where the script is located.
    """

    script_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = '/'.join(script_path.split('/')[:-1])
    return abs_path

if __name__ == "__main__":
    
    # retrieving the script arguments
    script_args = get_script_arguments()

    # Loading the data
    data_relative_path = script_args.data_relative_path
    data_path = os.path.join(os.getcwd(), data_relative_path)

    bbc_df = load_files( 
        container_path=data_path,
        encoding='UTF-8',
        decode_error='replace') 
    
    
   
    # preprocessing the text (bbc_df.data is a list)
    bbc_df.data = preprocess_data.clean_text_for_tfidf(
        data=bbc_df.data,
        normalize=True,
        remove_stopwords=True,
        remove_punct=True,
        language='english'
    )

    ## creating a preprocess pipeline
    # tfidf
    tfidf_vect = TfidfVectorizer(
        ngram_range=(1,1),
        max_df=0.95,
        min_df=2
    )
    
    # svd
    svd_dec = TruncatedSVD(
        n_components=1000,
        n_iter=30,
        random_state=script_args.random_state
    )

    lr = LogisticRegression()
    preprocess_pipe = [
        ('tfidf', tfidf_vect), ('svd', svd_dec), ('classifier', lr)]

    # pipe
    pipe = Pipeline(steps=preprocess_pipe)

    # Cross validation
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=script_args.random_state)
    
    results = cross_validate(
        estimator=pipe,
        X=bbc_df.data,
        y=bbc_df.target,
        cv=sss,
        n_jobs=-1,
    )

    print(results)





    





    