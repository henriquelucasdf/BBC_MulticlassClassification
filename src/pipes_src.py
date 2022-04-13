from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier


def get_preprocess_list(random_state=1):
    """
    It returns a list of tuples, where each tuple is a name and a scikit-learn preprocessor.
  
    Args:
      random_state: This is the seed for the random number generator. Defaults to 1
  
    Returns:
      A list of tuples.
    """
    
    tfidf_vect = TfidfVectorizer(
        ngram_range=(1,1),
        max_df=0.95,
        min_df=2
    )
    
    # Truncated SVD 
    svd_dec = TruncatedSVD(
        n_components=1000,
        n_iter=30,
        random_state=random_state
    )

    return [('tfidf', tfidf_vect), ('svd', svd_dec)]

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
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=7,
        random_state=random_state,
        n_jobs=n_jobs
    )

    return [('LogReg', lr), ('RandomForest', rf), ('LightGBM', lgb)]

