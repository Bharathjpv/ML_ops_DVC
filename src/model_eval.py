from pyexpat import model
import numpy as np
import pandas as pd
import os
import pickle
import json

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_processed_data(file_path):
    try:
        train_data = pd.read_csv(os.path.join(file_path, 'train_processed.csv'))
        test_data = pd.read_csv(os.path.join(file_path, 'test_processed.csv'))
        return train_data, test_data
    except Exception as e:
        raise Exception(f"Error loading processed data: {e}")

def load_model(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def evaluate_model(model, test_data) -> dict:
    try:
        X_test = test_data.drop('Potability', axis=1)
        y_test = test_data['Potability']
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")
    

def save_metrics(metrics_dict, metrics_path):
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f)
    except Exception as e:
        raise Exception(f"Error saving metrics: {e}")
    
def main():
    try:
        processed_data_path = os.path.join('data', 'processed')
        model_path = 'model.pkl'
        metrics_path = "metrics.json"

        _, test_data = load_processed_data(processed_data_path)
        model = load_model(model_path)
        metrics_dict = evaluate_model(model, test_data)
        save_metrics(metrics_dict, metrics_path)

        print("Model evaluation completed successfully.")
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")
    
if __name__ == "__main__":
    main()