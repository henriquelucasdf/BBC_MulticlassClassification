# BBC_MulticlassClassification

A Multiclass classification project using the BBC Dataset.

The product of this project is a machine learning pipeline capable of receiving news text, preprocessing it and classifying it into 5 categories: business, entertainment, politics, sport and tech.

## Introduction
On this project, I aimed to build a Continuous Training (CT) pipeline for further use in other projects. 

The outputs of the training pipeline (`main.py` script) are a serialized scikit-learn Pipeline, saved on the folder "models" and a text file containing the training logs, saved on the folder "logs".

The training pipeline consists in 3 steps:
1. **Preprocessing**: will load and preprocess the data, using a Custom Sklearn transformer to clean the text, a TF-IDF Vectorizer and a Dimensionality Reduction technique (TruncatedSVD).
2. **Models training**: this step will train and cross-validate 3 models (Logistic Regression, RandomForest and LightGBM) using a Stratified Shuffle Split (5 folds).
3. **Hyperparameter Optimization**: Using the best cross-validated model, this step will use a Randomized Search CV to find better hyperparameters for the model. 
4. **Post-processing**: This step will only save the best scikit-learn pipeline

It's expected that the Multiclass dataset is saved within the project folder, but the relative localization can be defined using the parameter `data_relative_path` on the main script.
        
    This project is not intended to be a demonstration of Data Science capabilities, but rather for software and machine learning engineering skills, as clean code, abstractions, and etc. Therefore, there are several models and preprocessing steps that could be tested but it is not my main objective.

## About the code
On the "src" folder, there are 4 scripts:
- `main.py`: responsible for orchestrate the whole pipeline
- `preprocess_src.py`: classes and functions for preprocessing. Currently there is only one class on this script (CustomTextPreprocessor, a custom sklearn transformer responsible for the cleaning of the input text).
- `pipe_definition.py`: in this script, we'll define the parameters of the pipeline steps, like ngram_range of TfidfVectorizer, max_depth of the RandomForestClassifier, ect. Beyond that, we can also define the possible hyperparameters that will be used on the Random Search. 
- `utils.py`: this one contains some auxiliary functions that are used on the other scripts. 

To run the pipeline with default values, simply execute:

`python3 main.py`

To use the parameters (example):

`python3 main.py --data_relative_path data/ --random_state 1 --cv_splits 5 --verbose 2 --best_model_metric balanced_accuracy`

This parameters do the following:
- `data_relative_path`: the name of the folder containing the dataset
- `random_state`: the random state for the models, data splitting and hyperparameters search.
- `cv_splits`: number of folders to cross-validate
- `verbose`: level of verbose for the cross-validation and hyperparameters search. The higher, the more messages.
- `best_model_metric`: the metric to define the best model after the cross-validation. Must use the sklearn [syntax](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

## Results and lessons
- Training on the BBC Dataset, I was able to get balanced accuracy, precision and recall greaters than 98%. 
- In my opinion, the code is modularized and abstracted enough to be deployed in an MLOps system, serving as part of a continuous training (CT) pipeline.  
- The use of logs is way better than a simple "print" statements.
- Implementing a Custom Sklearn transformer class proved to be a great lesson, showing the advantages of using the Sklearn Pipeline. 

## Technical Debts
- The experiments could be tracked using MLFlow
- The logging should be improved
- More of the pipeline components could be customizable via script parameters.


## References
- BBC Dataset:
    
    D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
    
    Available in this [link](http://mlg.ucd.ie/datasets/bbc.html).
