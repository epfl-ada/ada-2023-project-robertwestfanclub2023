import pandas as pd

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

    df_movie['Release Year'] = df_movie['Release Year'].astype('Int64')
    df_movie['Release Month'] = df_movie['Release Month'].astype('Int64')
    df_movie['Release Day'] = df_movie['Release Day'].astype('Int64')

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
            dict = row.to_dict()
            dict['Movie genres'] = [genre]
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

def filter_movies_by_genres(df, selected_genres):
    """
    Filter a DataFrame of movies based on selected genres.

    Args:
        df (pandas.DataFrame): The DataFrame containing the movie data.
        selected_genres (list): A list of genres to filter by.

    Returns:
        pandas.DataFrame: A filtered DataFrame containing only rows with selected genres.
    """
    # Use the .isin() method to filter rows where 'Movie genres' match the selected genres
    df_filter = df[df['Movie genres'].isin(selected_genres)]

    return df_filter
