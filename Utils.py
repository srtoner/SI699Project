import pandas as pd
import numpy as np
import pickle as pkl
import json
import os
from sklearn.model_selection import train_test_split
# How to use this file:
# Set your repository directory file
# in config.json in the base directory

# Run this script, which will then copy the config.json
# to all child directories

# You will then use the "load_file" and 
# "write_file" function for file IO to the 
# right directories

def load_file(filename, format, filedir = None):
    cwd = os.getcwd()
    if filedir:
        os.chdir(filedir)
    if format == 'csv':
        file = pd.read_csv(filename, encoding='utf-8')
    elif format == 'json':
        with open(filename, 'r') as f:
            file = json.load(f)
    elif format == 'pkl':
        with open(filename, 'rb') as f:
            file = pkl.load(f)
    os.chdir(cwd)
    return file


def write_file(file, filename, format, filedir = None):
    cwd = os.getcwd()
    if filedir:
        os.chdir(filedir)
    if format == 'csv':
        # assumes file is pandas dataframe
        file.to_csv(filename, index=False)
    elif format == 'json':
        with open(filename, 'w') as f:
            json.dump(f, file)
    elif format == 'pkl':
        with open(filename, 'wb') as f:
            pkl.dump(f, file)
    os.chdir(cwd)

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)

    repo_dir = config['REPODIR']

    for file in os.scandir():
        if file.is_dir():
            os.chdir(file.path)
            with open('config.json', 'w') as f:
                json.dump(config, f)
            os.chdir(repo_dir)

def split_data(data, features_col, target_col, test_size, val_size, random_state=42):

    # Split data into X (features) and y (target variable)
    X = data[features_col]
    y = data[target_col]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)

    # Split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size),
                                                      random_state=random_state,
                                                      stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
