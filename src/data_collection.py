from curses import raw
from genericpath import exists
from json import load
from sys import exc_info
from tkinter import E
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

def load_params(config_path: str) -> dict:
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

config = load_params("params.yaml")

test_size=config['data_collection']['test_size']

def load_data(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {path}: {e}")


def split_data(data: pd.DataFrame, test_size: float) -> tuple:
    try:
        train_data, test_data = train_test_split(data, test_size=test_size)
        return train_data, test_data
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except Exception as e:
        raise Exception(f"Error saving data: {e}")
    
def main():
    try:
        data_file_path = 'data/water_potability.csv'
        params_file_path = 'params.yaml'
        raw_data_path = os.path.join('data', 'raw')

        data = load_data(data_file_path)
        params = load_params(params_file_path)
        train_data, test_data = split_data(data, test_size=params['data_collection']['test_size'])
        save_data(train_data, test_data, raw_data_path)
    except Exception as e:
        raise Exception(f"Error in data collection process: {e}")
    
if __name__ == "__main__":
    main()