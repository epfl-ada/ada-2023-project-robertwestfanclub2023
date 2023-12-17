import pandas as pd
from scipy.stats import ttest_ind, pearsonr
import numpy as np

def save_dataframe_to_csv(dataframe, file_path):
    """
    Saves a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame will be saved.
        
    Returns:
        bool: True if the DataFrame was successfully saved, False otherwise.
    """
    try:
        dataframe.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving DataFrame to {file_path}: {str(e)}")
        return False


def get_path(url):
    """
    Returns data path for input url.
    Returned path can be used to make dataframe.

    Parameters
    ----------
    url: string
        The url you desire to find path for

    Returns
    -------
    path: string
        The path which can be used to make dataframe in pandas

    """
    return "https://drive.google.com/uc?id=" + url.split("/")[-2]

def top_k_with_oscars(df, k, feature, ascending=False):
    """
    Returns top k rows of a feature from a movie with an Oscars.

    Parameters
    ----------
    df: pandas dataframe
        The dataframe you want to find top k rows from
    k: int
        The number of rows you want to return
    feature: string
        The column you want to find top k rows from

    Returns
    -------
    df: pandas dataframe
        The dataframe with top k rows with most oscars in column

    """
    # Filter the dataset for movies that have won an Oscar
    winning_movies = df[df['Winner'] == 1]

    sorted = winning_movies.sort_values(feature, ascending=ascending).head(k)
    for index, movie in sorted.iterrows():
        print(f"Movie: {movie['Movie name']}")
        print(f"Year: {movie['Movie Year']}")
        print(f"Category: {movie['Category']}")
        print(f"{feature}: {movie[feature]:.2f}")
        print("---")

def t_test_and_correlation_on_winner_vs_col(df, column):
    """
    Perform a t-test and correlation analysis on a column of a DataFrame compared to its oscars win.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The name of the column to analyze.

    Returns:
    None
    """
        
    # Select relevant columns for the analysis
    columns_to_test = [column, 'Winner']

    # Create a subset DataFrame with the selected columns
    df_test = df[columns_to_test].copy()

    # Convert 'Winner' column to numeric (True/False to 1/0)
    df_test['Winner'] = df_test['Winner'].astype(int)

    # Remove rows with missing or infinite values
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.dropna(subset=[column])

    # Split the data into two groups based on Oscar win
    winner = df_test[df_test['Winner'] == 1][column]
    loser = df_test[df_test['Winner'] == 0][column]

    # Perform the independent two-sample t-test
    statistic, p_value = ttest_ind(winner, loser, nan_policy='omit')

    # Perform correlation analysis between box office revenue and the number of Oscars won
    correlation_coefficient, correlation_p_value = pearsonr(df_test[column], df_test['Winner'])

    # Display the statistical test and correlation values
    print(f'T-test statistic: {statistic}')
    print(f'T-test p-value: {p_value}')
    print(f'Correlation coefficient: {correlation_coefficient}')
    print(f'Correlation p-value: {correlation_p_value}')
