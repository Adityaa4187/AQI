import pandas as pd
from scipy.stats import zscore

def load_data(file_path):
    """
    Load dataset from the given CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Remove rows with missing values
    return df

def remove_outliers_z_transform(df, columns, threshold=3):
    """
    Remove outliers from the dataframe based on Z-scores.
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
