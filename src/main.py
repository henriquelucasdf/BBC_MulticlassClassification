import os
import pytz
import logging
import argparse
import numpy as np
from time import time
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate

import utils
import pipe_definition


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
    preprocess_list = pipe_definition.get_preprocess_list(
        random_state=script_args.random_state)

    logging.info(
        f"   The defined preprocess steps are:   {preprocess_list}")

    ## Getting the models pipeline list for the Pipeline
    models_list = pipe_definition.get_models_list(
        random_state=script_args.random_state,
        n_jobs=-1)

    logging.info(
        f"   The defined models to test are:   {models_list}")

    # Cross validation of models
    sss = StratifiedShuffleSplit(
        n_splits=script_args.cv_splits,
        test_size=0.2,
        random_state=script_args.random_state)

    # training and evaluating each model   
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
        pipe_start_params = pipe.get_params()

        # Initializing the cross-validation
        results = cross_validate(
            estimator=pipe,
            X=bbc_df.data,
            y=bbc_df.target,
            cv=sss,
            scoring=['balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            return_train_score=True,
            n_jobs=-1,
            verbose=script_args.verbose
        )

        # logging the test metrics
        logging.info(
            f"    Finishing the cross-validation of the model {model_name} after {time() - start_time :.2f} seconds...")

        test_results = {str(key)[5:]: np.round(np.mean(value),3) for key, value in results.items() if 'test_' in key}
        logging.info(f"    {model_name} - Mean CV Results: {test_results}")

        # storing {model_name: test_results}
        models_kpis_dict[model_name] = test_results

    
    del pipe # deletes last pipeline
    # After evaluating all models > Hiperparameters optimization of the best model
    best_model_tuple = utils.get_best_model(
        kpis_dict=models_kpis_dict,
        models_list=models_list,
        metric_for_eval=script_args.best_model_metric
    )

    ## Hyperparameters optimization
    logging.info("    Starting the hyperparameters optimization...")
    
    # pipeline steps
    hyper_steps = preprocess_list.copy()
    hyper_steps.append(best_model_tuple)

    hyper_pipe = Pipeline(steps=hyper_steps)

    # getting the possible hyperparameters 
    hyper_params = pipe_definition.get_hiperparameters_values(pipeline=hyper_pipe)

    # CV for hyperparameters
    sss_hyper = StratifiedShuffleSplit(
        n_splits=3,
        test_size=0.25,
        random_state=script_args.random_state
    )

    # Random Search
    rs = RandomizedSearchCV(
        estimator=hyper_pipe,
        param_distributions=hyper_params,
        n_iter=10,
        n_jobs=-1,
        refit=True,
        cv=sss_hyper,
        verbose=script_args.verbose,
        scoring=script_args.best_model_metric,
        random_state=script_args.random_state
    )

    rs.fit(X=bbc_df.data, y=bbc_df.target)

    # Loggings
    logging.info(f"    The best parameters found were: {rs.best_params_}")
    logging.info(f"    The best metric ({script_args.best_model_metric}) was: {rs.best_score_}")

    # Saving the best estimator:
    logging.info("    Saving the best estimator on the folder 'models'...")
    utils.save_models(
        estimator=rs.best_estimator_,
        name='best_pipeline',
        folder='../models')
    
    logging.info("    Finalizing the script...")


    





    





    