import pandas as pd
from scipy.stats import zscore

def load_data(file_path):
    """
    Load dataset from the given CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the loaded data (nulls retained for post-split removal).
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df

def remove_nulls(df):
    """
    Drop rows with missing values.
    Should be called on the training set only, after the train/val/test split.
    :param df: The input DataFrame.
    :return: DataFrame with null rows removed.
    """
    return df.dropna()

def remove_outliers_z_transform(df, columns, threshold=6):
    """
    Remove outliers from the dataframe based on Z-scores.
    Should be called on the training set only, after the train/val/test split.
    :param df: The input DataFrame.
    :param columns: List of columns to check for outliers.
    :param threshold: Z-score threshold to define outliers.
    :return: DataFrame after removing outliers.
    """
    for col in columns:
        df['z_score'] = zscore(df[col])
        df = df[df['z_score'].abs() <= threshold]
        df = df.drop(columns=['z_score'])
    return df