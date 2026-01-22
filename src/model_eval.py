from pyexpat import model
import numpy as np
import pandas as pd
import os
import pickle
import json

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = pd.read_csv(os.path.join('data', 'processed', 'test_processed.csv'))

X_test = test_data.drop('Potability', axis=1)
y_test = test_data['Potability']

model = pickle.load(open('model.pkl', 'rb'))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}

with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f)