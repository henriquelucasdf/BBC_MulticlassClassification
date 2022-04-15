import os
import pytz
import logging
import argparse
import numpy as np
from time import time
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files

from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

import utils
import pipes_src


if __name__ == "__main__":

    # configurating the logging
    utils._config_logging()
    
    # retrieving the script arguments
    script_args = utils._get_script_arguments()
    logging.info(
        f"   A new run has started with the arguments: {script_args}")

    ## Loading the data
    data_relative_path = script_args.data_relative_path
    data_path = os.path.join(os.getcwd(), data_relative_path)

    bbc_df = load_files( 
        container_path=data_path,
        encoding='UTF-8',
        decode_error='replace') 
     
    ## Creating a preprocess list for the Pipeline: contains a text preprocessor, a tfidf and an TruncatedSVD
    preprocess_list = pipes_src.get_preprocess_list(
        random_state=script_args.random_state)

    logging.info(
        f"   The defined preprocess steps are:   {preprocess_list}")
    
    ## Getting the models pipeline list for the Pipeline
    models_list = pipes_src.get_models_list(
        random_state=script_args.random_state,
        n_jobs=-1)
    
    logging.info(
        f"   The defined models to test are:   {models_list}")

    # Cross validation of models
    
    models_kpis_dict = {}
    for model_tuple in models_list:        
        start_time = time()

        # name of the model
        model_name = model_tuple[0]
        logging.info(
            f"    Starting the cross-validation of the model {model_name}...")

        # creating the final steps with the model
        final_list = preprocess_list.copy()
        final_list.append(model_tuple)
    
        # pipe
        pipe = Pipeline(steps=final_list)

        # Cross validation
        sss = StratifiedShuffleSplit(
            n_splits=2, #script_args.cv_splits,
            test_size=0.2,
            random_state=script_args.random_state)
        
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

        # logging the test metrics
        logging.info(
            f"    Finishing the cross-validation of the model {model_name} after {time() - start_time :.2f} seconds...")

        test_results = {str(key)[5:]: np.round(np.mean(value),3) for key, value in results.items() if 'test_' in key}
        logging.info(f"    {model_name} - Mean CV Results: {test_results}")

        # storing {model_name: test_results}
        models_kpis_dict[model_name] = test_results

    # After evaluating all models > Hiperparameters optimization of the best model
    best_model_tuple = utils.get_best_model(
        kpis_dict=models_kpis_dict,
        models_list=models_list,
        metric_for_eval='balanced_accuracy'
    )

    logging.info("Finalizing the script...")


    





    





    