from re import X
import pandas as pd
import numpy as np
import os
import yaml

import pickle
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

n_estimators = config['model_building']['n_estimators']

train_data = pd.read_csv(os.path.join('data', 'processed', 'train_processed.csv'))
test_data = pd.read_csv(os.path.join('data', 'processed', 'test_processed.csv'))

X_train = train_data.drop('Potability', axis=1)
y_train = train_data['Potability']

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))