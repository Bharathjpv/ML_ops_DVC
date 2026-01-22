import pandas as pd
import numpy as np
import os

train_data = pd.read_csv(os.path.join('data', 'raw', 'train.csv'))
test_data = pd.read_csv(os.path.join('data', 'raw', 'test.csv'))

def fill_missing_values_median(df):
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df

train_processed_data = fill_missing_values_median(train_data)
test_processed_data = fill_missing_values_median(test_data)

data_path = os.path.join('data', 'processed')

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
