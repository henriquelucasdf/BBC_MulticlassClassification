import os
import argparse
import numpy as np

from sklearn.datasets import load_files

def get_script_arguments():

    # Initializing
    parser = argparse.ArgumentParser()

    # adding arguments
    parser.add_argument('--data_relative_path', type=str, default='data/')


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

    bbc_df = load_files(data_path, encoding='utf-8', decode_error='replace')

    