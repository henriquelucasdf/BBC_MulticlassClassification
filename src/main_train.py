import os
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

import preprocess_data, pipes_src

def get_script_arguments():

    # Initializing
    parser = argparse.ArgumentParser()

    # adding arguments
    parser.add_argument('--data_relative_path', type=str, default='data/')
    parser.add_argument('--random_state', type=int, default=1)

    # parsing
    script_args = parser.parse_args()

    return script_args


if __name__ == "__main__":
    
    # retrieving the script arguments
    script_args = get_script_arguments()

    ## Loading the data
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

    ## Creating a preprocess list for the Pipeline: contains a tfidf and an TruncatedSVD
    preprocess_list = pipes_src.get_preprocess_list(
        random_state=script_args.random_state)

    print(f"Preprocess steps: {preprocess_list}")
    
    ## Getting the models pipeline list for the Pipeline
    models_list = pipes_src.get_models_list(
        random_state=script_args.random_state,
        n_jobs=-1)
    
    print(f"Models List: {models_list}")  

    for model_tuple in models_list:
        # name of the model
        model_name = model_tuple[0]
        print(f"\nStarting the cross-validation of the model {model_name}...")

        final_list = preprocess_list.copy()
        final_list.append(model_tuple)
    
        # pipe
        pipe = Pipeline(steps=final_list)

        # Cross validation
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=script_args.random_state)
        
        # Initializing the cross-validation
        results = cross_validate(
            estimator=pipe,
            X=bbc_df.data,
            y=bbc_df.target,
            cv=sss,
            scoring=['balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            return_train_score=True,
            n_jobs=-1,
        )

        print(f"{model_name} - CV Mean Balanced Accuracy: {np.mean(results['test_balanced_accuracy']):.3f}")
        print(f"{model_name} - CV Mean Precision Macro: {np.mean(results['test_precision_macro']):.3f}")
        print(f"{model_name} - CV Mean Recall Macro: {np.mean(results['test_recall_macro']):.3f}")
        print(f"{model_name} - CV Mean F1 Macro: {np.mean(results['test_f1_macro']):.3f}")
    
    print("Finalizing the script...")


    





    





    