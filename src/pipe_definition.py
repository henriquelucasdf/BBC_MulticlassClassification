import logging
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess_src import CustomTextPreprocessor


def get_preprocess_list(random_state=1):
    """
    It returns a list of tuples, where each tuple is a name and a scikit-learn preprocessor.
  
    Args:
      random_state: This is the seed for the random number generator. Defaults to 1
  
    Returns:
      A list of tuples.
    """
    # text preprocessor
    tp = CustomTextPreprocessor(
        normalize=True,
        remove_punct=True,
        remove_stopwords=True,
        language='english'
    )
    # TFIDF Vectorizer
    tfidf_vect = TfidfVectorizer(
        ngram_range=(1,1),
        max_df=0.95,
        min_df=2,
        max_features=5000
    )
    
    # Truncated SVD 
    svd_dec = TruncatedSVD(
        n_components=1000,
        n_iter=30,
        random_state=random_state
    )

    return [('text_proc', tp),('tfidf', tfidf_vect), ('svd', svd_dec)]

def get_models_list(random_state=1, n_jobs=-1):
    """
    It returns a list of tuples, where each tuple contains the name of the model and the model itself
    
    Args:
      random_state: This is the seed used by the random number generator. It can be any integer.
    Defaults to 1
      n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    
    Returns:
      A list of tuples.
    """

    # Logistic Regression
    lr = LogisticRegression(
        random_state=random_state,
        n_jobs=n_jobs)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        random_state=random_state,
        n_jobs=n_jobs
    )

    # GB classifier
    lgb = LGBMClassifier(
        learning_rate=0.05,
        n_estimators=500,
        max_depth=5,
        random_state=random_state,
        n_jobs=n_jobs
    )

    return [('LogReg', lr), ('RandomForest', rf), ('LightGBM', lgb)]


def get_possible_hiperparameters(steps_name):
    """
    It takes a list of steps in a pipeline and returns a dictionary of hyperparameters that can be used
    in a RandomizedSearchCV
    
    Args:
      steps_name: the names of the steps in the pipeline
    
    Returns:
      A dictionary with the hyperparameters for the models in the pipeline.
    """
    
    # Define the general hyperparameters
    params_dict = {
        # Preprocessors
        'tfidf': {
            'ngram_range': [(1,1), (1,2), (2,2)],
            'max_df': [0.9, 0.95, 0.99],
            'min_df': [1, 3, 5],
            'max_features': [3000, 5000, 7500, 10000]
        },
        'svd': {
            'n_components': [500, 750, 1000],
            'n_iter': [30,50, 75],
        },
        # Models
        'LogReg': {
            'C': [0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'penalty': ['none', 'l2'],
            'max_iter': [100, 200, 300]
        },
        'RandomForest': {
            'n_estimators': [100, 500, 1000, 1500],
            'max_features': [1,5,10,20],
            'max_depth': [3, 5, 7],
        },
        'LightGBM': {
            'lambda_l1': [0, 0.5, 1, 10,100],
            'lambda_l2': [0, 0.5, 1, 10,100],
            'bagging_fraction': [0, 0.25, 0.5, 0.75, 1],
            'bagging_freq': [0, 5, 10, 50],
            'max_depth': [2, 4, 6, 8],
            'num_iterations': [100, 500, 1000, 1500],
            'learning_rate': [0.05, 0.1, 0.25, 0.5]
        }
    }

    # check which keys are in the pipeline steps
    params_in_pipe = {key: value for key, value in params_dict.items() if
        key in steps_name}

    # formatting the names of the parameters
    params_in_pipe = {f'{keys1}__{keys2}': values2 for keys1, values1 in params_in_pipe.items()
        for keys2, values2 in values1.items()}
    
    return params_in_pipe
    

def get_hiperparameters_values(pipeline):
    """
    It returns a dictionary with the possible hiperparameters for each step of the pipeline
    
    Args:
      pipeline: the pipeline object
    
    Returns:
      A dictionary with the possible hiperparameters for each step of the pipeline.
    """

    # steps names
    steps_names = list(pipeline.named_steps.keys())

    # params
    params_dict = get_possible_hiperparameters(steps_names)
    logging.info(f"    The possible hyperparameters are: {params_dict}")

    return params_dict



