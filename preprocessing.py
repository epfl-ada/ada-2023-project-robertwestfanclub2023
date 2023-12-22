
####################################################################################################
# IMPORTS
####################################################################################################

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Third-party libraries
import tqdm
from tqdm import tqdm
import pycountry_convert as pc
from scipy.stats import ttest_ind, pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose

####################################################################################################
# Data Cleaning and Preprocessing
####################################################################################################


####################
# Utils functions
####################

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
    
####################
# Preprocessing functions
####################

def preprocess_dict(row):
    """
    Preprocesses a row of movie data.

    Args:
        row (pd.Series): A row of movie data as a Pandas Series.

    Returns:
        pd.Series: The preprocessed row with updated values for 'Movie countries',
                   'Movie languages', and 'Movie genres'.
    """
    movie_countries = list(eval(row['Movie countries']).values())
    movie_languages = list(eval(row['Movie languages']).values())
    movie_genres = list(eval(row['Movie genres']).values())

    # Replace empty lists with [None]
    if not movie_countries:
        movie_countries = [None]

    if not movie_languages:
        movie_languages = [None]
    else:
        # Check for 'Silent film' and process other languages
        if 'Silent film' in movie_languages:
            movie_languages = ['Silent film']
        else:
            processed_languages = []
            for language in movie_languages:
                if 'Language' in language:
                    processed_languages.append(language.replace('Language', '').strip())
                else:
                    processed_languages.append(language)
            movie_languages = processed_languages

    # Replace empty lists with [None]
    if not movie_genres:
        movie_genres = [None]

    # Update the row with preprocessed values
    row['Movie countries'] = movie_countries
    row['Movie languages'] = movie_languages
    row['Movie genres'] = movie_genres

    return row

def update_release_month_and_day(df):
    """
    Update the 'Release Month' and 'Release Day' columns in a DataFrame based on the 'Movie release date' column.

    This function iterates through the DataFrame and checks if 'Month Available' is False and 'Movie release date' is a string.
    If both conditions are met, it extracts the year, month, and day from 'Movie release date' and updates the corresponding columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to update in-place.

    Returns:
        None
    """
    for index, row in df.iterrows():
        if not row['Month Available'] and isinstance(row['Movie release date'], str): 
            year, month, day = row['Movie release date'].split('-')  
            if int(year) == row['Release Year']:
                df.at[index, 'Release Month'] = int(month)
                df.at[index, 'Release Day'] = int(day)
                df.at[index, 'Month Available'] = True
                
    return df

def extract_date_info_and_update_df(df_movie):
    """
    Extract year, month, and day information from a date column in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a date column to extract information from.

    Returns:
        pandas.DataFrame: The DataFrame with 'Release Year', 'Release Month', and 'Release Day' columns added.
    """
    # Define the function to extract date information from a string
    def extract_date_info(date_str):
        month_flag = False # Flag to detect if a month is present
        if not isinstance(date_str, str):
            return None, None, None, month_flag
        else:
            parts = date_str.split('-')
            parts = [int(part) for part in parts]
            if len(parts) >= 2:
                month_flag = True
            while len(parts) < 3:
                parts.append(None)
            return parts[0], parts[1], parts[2], month_flag

    # Apply the function to the date column and store the results in a temporary column
    df_movie['Release Date Info'] = df_movie['Movie release date'].apply(extract_date_info)

    # Split the 'Release Date Info' column into three distinct columns: 'Release Year', 'Release Month', 'Release Day'
    df_movie[['Release Year', 'Release Month', 'Release Day', 'Month Available']] = pd.DataFrame(df_movie['Release Date Info'].tolist(), index=df_movie.index)

    release_list = df_movie['Release Year'].to_list()
    release_list =  [int(x) if not pd.isna(x) else -1 for x in release_list]
    df_movie['Release Year'] = release_list


    release_list = df_movie['Release Month'].to_list()
    release_list =  [int(x) if not pd.isna(x) else -1 for x in release_list]
    df_movie['Release Month'] = release_list

    release_list = df_movie['Release Day'].to_list()
    release_list =  [int(x) if not pd.isna(x) else -1 for x in release_list]
    df_movie['Release Day'] = release_list

    # Drop the temporary 'Release Date Info' and 'Movie release date' columns
    df_movie.drop(columns=['Release Date Info', 'Movie release date'], inplace=True)

    return df_movie

def process_movie_data(df_movie, save_to_path=None):
    """
    Process a movie dataset DataFrame and extract information about countries, genres, languages, and release dates.

    Parameters:
    - df_movie (pd.DataFrame): The DataFrame containing the movie dataset.
    - save_to_path (str, optional): The path to save the processed DataFrame as a CSV file (default: None).

    Returns:
    - df_process (pd.DataFrame): A DataFrame containing processed movie data.
    """

    # Initialize lists and dictionaries to store data
    dict_countries = set()
    dict_genre = set()
    dict_languages = set()

    df_process = pd.DataFrame(columns=df_movie.columns)

    # Iterate through each row in the DataFrame
    for index, row in tqdm(df_movie.iterrows(), total=len(df_movie)):
        # Update sets with data from the current row
        dict_countries.update(eval(row['Movie countries']))
        dict_genre.update(eval(row['Movie genres']))
        dict_languages.update(eval(row['Movie languages']))

        # Preprocess data and append to lists
        row = preprocess_dict(row)
        for genre in row['Movie genres']:
            for country in row['Movie countries']:
                for language in row['Movie languages']:
                    dict = row.to_dict()
                    dict['Movie genres'] = [genre]
                    dict['Movie countries'] = [country]
                    dict['Movie languages'] = [language]
                    df_process = pd.concat([df_process, pd.DataFrame(dict)], ignore_index=True)

    # Print the number of unique countries, languages, and genres
    print(f'In the movie dataset, there are {len(dict_countries)} different countries.')
    print(f'In the movie dataset, there are {len(dict_languages)} different languages.')
    print(f'In the movie dataset, there are {len(dict_genre)} different genres.')

    # Extract date information from the 'Movie release date' column and update the DataFrame
    df_process = extract_date_info_and_update_df(df_process)

    # Optionally save the processed DataFrame to a specified path
    if save_to_path:
        df_process.to_csv(save_to_path, index=False)

    return df_process

def country_to_continent(country_name):
    """
    Converts a country name to its continent name.

    Args:
        country_name (str): The name of the country.

    Returns:
        str or np.nan: The continent name if the country is found, or np.nan if not found.
    """
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    except KeyError:
        return np.nan
    
def assign_movie_continents(df):
    """
    Assign movie continents to a DataFrame based on the 'Movie countries' column.

    Args:
        df (pandas.DataFrame): The DataFrame containing movie data.

    Returns:
        pandas.DataFrame: The DataFrame with an added 'Movie Continent' column.
    """
    list_continents = []

    # Iterate through DataFrame rows and assign continents
    for index, row in tqdm(df.iterrows()):
        if pd.isna(row['Movie countries']):
            list_continents.append(np.nan)
        else:
            list_continents.append(country_to_continent(row['Movie countries']))

    # Add the 'Movie Continent' column to the DataFrame
    df['Movie Continent'] = list_continents

    return df

####################################################################################################
# Research Questions 1
####################################################################################################

#########################################
# Exploratory and preprocessing functions
#########################################

def count_and_sort_movie_genres(df_movie):
    """
    Count and sort movie genres from a DataFrame.

    Parameters:
    - df_movie (pd.DataFrame): DataFrame containing the movie dataset.

    Returns:
    - sorted_genre_counts (dict): A dictionary containing genres as keys and their counts as values,
      sorted in descending order by count.
    """

    # Initialize an empty dictionary to store genre counts
    genre_counts = {}

    # Extract the 'Movie genres' column and convert it to a list
    list_genre = df_movie['Movie genres'].to_list()

    # Count the occurrences of each genre
    for genre in list_genre:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

    # # Sort the genre counts dictionary by count in descending order
    sorted_genre_counts = dict(sorted(genre_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_genre_counts

####################################################################
# Comprehensive Seasonality Analysis Across All Genres and Locations
####################################################################

def calculate_canova_hansen_test(df_year, ch_test):
    """
    Calculate the Canova-Hansen Testing results for seasonal differences
    for each unique genre-continent combination in the given DataFrame.

    Args:
    df_year (pd.DataFrame): The DataFrame containing movie data.
    ch_test: The Canova-Hansen test object.

    Returns:
    pd.DataFrame: A DataFrame with columns 'Genre', 'Continent', and 'D Value' representing the genre,
    continent, and estimated seasonal differencing term, respectively.
    """
    # Initialize empty lists to store the results
    genres_list = []
    continents_list = []
    D_values_list = []

    df = df_year.copy()
    
    # Get unique genres and continents in your DataFrame
    unique_genres = df['Movie genres'].unique()
    unique_continents = df['Movie Continent'].unique()

    # Iterate through unique genre-continent combinations
    for genre in unique_genres:
        for continent in unique_continents:
            subset = df[(df['Movie genres'] == genre) & (df['Movie Continent'] == continent)]
            data = subset['Percentage'].values

            D = ch_test.estimate_seasonal_differencing_term(data)

            # Store the results in the lists
            genres_list.append(genre)
            continents_list.append(continent)
            D_values_list.append(D)

    # Create a DataFrame from the lists
    results_df = pd.DataFrame({
        'Genre': genres_list,
        'Continent': continents_list,
        'D Value': D_values_list
    })

    return results_df

def perform_seasonal_decomposition(df):
    """
    Perform seasonal decomposition on movie data.

    Parameters:
        df (DataFrame): A pandas DataFrame containing movie data with columns including 'Release Month',
                        'Movie genres', 'Movie Continent', 'Counts', 'Percentage', and 'Release Year'.

    Returns:
        DataFrame: A DataFrame containing seasonal decomposition results including 'Release Month', 'Movie genres',
                   'Movie Continent', 'Counts', 'Percentage', and 'Seasonality'.

    """
    decomposition_results = pd.DataFrame(columns=['Release Month', 'Movie genres', 'Movie Continent', 'Counts', 'Percentage', 'Seasonality'])

    for continent in df['Movie Continent'].unique():
        for genre in df['Movie genres'].unique():
            # Filter data by continent, genre, and release year
            data = df[(df['Movie genres'] == genre) & (df['Movie Continent'] == continent) & (df['Release Year'] >= 2000)]

            # Perform seasonal decomposition on the 'Percentage' column
            decomposition = seasonal_decompose(data['Percentage'], model='multiplicative', period=12)
            
            # Create a DataFrame to store decomposition results
            decomposition_df = pd.DataFrame({
                'Release Month': data['Release Month'],
                'Movie genres': genre,
                'Movie Continent': continent,
                'Counts': data['Counts'],
                'Percentage': data['Percentage'],
                'Seasonality': decomposition.seasonal,
            })

            # Concatenate decomposition results to the final DataFrame
            decomposition_results = pd.concat([decomposition_results, decomposition_df], axis=0)

    return decomposition_results

####################################################################################################
# Research Questions 2
####################################################################################################

