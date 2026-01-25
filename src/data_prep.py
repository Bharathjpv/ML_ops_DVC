import pandas as pd
import numpy as np
import os

from semver import process

def load_data(file_path):
    try:
        train_data = pd.read_csv(os.path.join(file_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(file_path, 'test.csv'))
        return train_data, test_data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def fill_missing_values_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")
    

def save_processed_data(train_processed_data, test_processed_data, file_path):
    try:
        os.makedirs(file_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(file_path, 'train_processed_mean.csv'), index=False)
        test_processed_data.to_csv(os.path.join(file_path, 'test_processed_mean.csv'), index=False)
    except Exception as e:
        raise Exception(f"Error saving processed data: {e}")
    


def main():
    try:

        raw_data_path = os.path.join('data', 'raw')
        processed_data_path = os.path.join('data', 'processed')

        train_data, test_data = load_data(raw_data_path)
        train_processed_data, test_processed_data = fill_missing_values_mean(train_data), fill_missing_values_mean(test_data)
        save_processed_data(train_processed_data, test_processed_data, processed_data_path)
        print("Data processing completed successfully.")
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()