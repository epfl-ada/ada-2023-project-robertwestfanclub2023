import pandas as pd

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