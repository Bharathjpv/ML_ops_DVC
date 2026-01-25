from genericpath import exists
from sys import exc_info
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

test_size=config['data_collection']['test_size']

data = pd.read_csv('data/water_potability.csv')

train_data, test_data = train_test_split(data, test_size=test_size)
data_path = os.path.join('data', 'raw')

os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)