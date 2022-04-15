import os
import pytz
import logging
import argparse
import numpy as np
from time import time
from datetime import datetime

def _get_script_arguments():
    """
    It creates a parser object, adds arguments to it, and then parses the arguments
    
    Returns:
      The script arguments.
    """

    # Initializing
    parser = argparse.ArgumentParser()

    # adding arguments
    parser.add_argument('--data_relative_path', type=str, default='data/')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--cv_splits', type=int, default=5)

    # parsing
    script_args = parser.parse_args()

    return script_args


def _get_datetime_as_string():
    """
    It returns a string with the current date and time in the format 'dd-mm-yyyy-hhmm'
    
    Returns:
      A string with the current date and time.
    """

    sp = pytz.timezone('America/Sao_Paulo')
    timestr = datetime.now().astimezone(sp)
    timestr = timestr.strftime('%d-%m-%Y-%Hh%M')

    return timestr

def _config_logging():
    """
    It creates a folder called 'logs' in the current working directory, and creates a file called
    'logs_<datetime>.txt' in that folder
    """
    
    # getting datetime
    time_str = _get_datetime_as_string()

    ## creating the log txt file
    # create the log folder
    logs_folder = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_folder, exist_ok=True)
    
    # create the file
    filepath = os.path.join(logs_folder, f'logs_{time_str}.txt')    
    with open(filepath, 'w') as fp:
        pass

    # config the logging 
    logging.basicConfig(filename=filepath, level=logging.DEBUG)

def get_best_model(kpis_dict, models_list, metric_for_eval='balanced_accuracy'):
    """
    It takes a dictionary of models and their metrics, and returns a tuple with the name  and the
    class of the model with the best metric
    
    Args:
      kpis_dict: a dictionary of dictionaries, where the keys are the names of the models, and the
    values are dictionaries of the metrics for each model.
      models_list: list of tuples containing the (model names, model_class)
      metric_for_eval: the metric to be used for model selection. Defaults to 'balanced_accuracy'
    
    Returns:
      The name of the best model and its class.
    """
    
    logging.info(f"    Getting the best model by the metric {metric_for_eval}")
    
    # getting only the metric of interest (value is also a dict)
    kpis_dict_clean = {
        key: value[metric_for_eval] for key, value in kpis_dict.items()
        }
    
    # name of the best model
    max_key = max(zip(kpis_dict_clean.values(), kpis_dict_clean.keys()))[1]

    # returning the (best_model_name, best_model_class) tuple
    best_model_tuple = [model_tuple for model_tuple in models_list if model_tuple[0]==max_key][0]
    logging.info(f"    Best model found: {best_model_tuple[0]}")
    
    return best_model_tuple