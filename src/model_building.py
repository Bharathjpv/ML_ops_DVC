from re import X
from networkx import config
import pandas as pd
import numpy as np
import os
import yaml

import pickle
from sklearn.ensemble import RandomForestClassifier

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        raise Exception(f"Error reading YAML file: {e}")
    

def load_processed_data(file_path):
    try:
        train_data = pd.read_csv(os.path.join(file_path, 'train_processed.csv'))
        test_data = pd.read_csv(os.path.join(file_path, 'test_processed.csv'))
        return train_data, test_data
    except Exception as e:
        raise Exception(f"Error loading processed data: {e}")

def train_model(train_data, n_estimators):
    try:
        X_train = train_data.drop('Potability', axis=1)
        y_train = train_data['Potability']

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")
    
def save_model(clf, model_path):
    try:
        pickle.dump(clf, open(model_path, 'wb'))
    except Exception as e:
        raise Exception(f"Error saving model: {e}")

def main():
    try:
        param_file_path = 'params.yaml'
        processed_data_path = os.path.join('data', 'processed')
        model_path = 'model.pkl'

        params = read_yaml(param_file_path)
        n_estimators = params['model_building']['n_estimators']

        train_data, _ = load_processed_data(processed_data_path)
        clf = train_model(train_data, n_estimators)
        save_model(clf, model_path)

        print("Model training and saving completed successfully.")
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")
    
if __name__ == "__main__":
    main()