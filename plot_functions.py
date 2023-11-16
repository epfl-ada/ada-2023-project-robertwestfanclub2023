import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import preprocessing
from preprocessing import filter_movies_by_genres

def plot_movie_release_distribution(df_movie):
    """
    Plots the distribution of movie releases by year within the range 1915-2012.

    Parameters:
    - df_movie (pd.DataFrame): DataFrame containing movie data.

    Returns:
    - None
    """
    # Create bins for the histogram
    bins = np.arange(1915, 2013) - 0.5
    
    # Create the histogram with log scale
    plt.hist(df_movie["Release Year"], histtype="step", bins=bins, log=True, color='b')
    
    # Set the plot title and labels
    plt.title("Distribution of Movie Releases by Year (1915-2012)")
    plt.xlabel("Year of Release")
    plt.ylabel("Number of Movies")

    # Customize the x-axis ticks
    plt.xticks(range(1915, 2013, 10), rotation=45)
    
    # Turn off the grid
    plt.grid(False)

    # Show the plot
    plt.show()


def create_genre_pie_chart(df_filter):
    """
    Create a pie chart to visualize the distribution of movie genres in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing movie genre data.

    Returns:
        None

    This function takes a DataFrame with a 'Movie genres' column and creates a pie chart to visualize
    the distribution of movie genres in the dataset.

    """
    labels = df_filter['Movie genres'].value_counts().index.tolist()
    data_counts = df_filter['Movie genres'].value_counts().values.tolist()

    # Create the pie chart using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.set_palette("pastel")
    plt.pie(data_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Movie Genres Distribution")

    plt.show()


def plot_month_release_availability(df):
    """
    Create a pie chart to visualize the proportion of Month Release Availability in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'Release Month' column.

    Returns:
        None
    """

    # Calculate the number of NaN and non-NaN values
    nan_count = df['Release Month'].isna().sum()
    non_nan_count = df['Release Month'].count()

    # Print the number of movies with missing release month and the total number of movies in the dataset
    print("In this dataset, there are {} movies with missing release month out of a total of {} movies.".format(nan_count, nan_count + non_nan_count))
    # Calculate the mean and standard deviation for 'Release Year' where 'Month Available' is True
    mean_true = df[df['Month Available']]['Release Year'].mean()
    std_true = df[df['Month Available']]['Release Year'].std()

    # Calculate the mean and standard deviation for 'Release Year' where 'Month Available' is False
    mean_false = df[~df['Month Available']]['Release Year'].mean()
    std_false = df[~df['Month Available']]['Release Year'].std()

    # Round the mean and standard deviation to integers
    mean_true = round(mean_true)
    std_true = round(std_true)
    mean_false = round(mean_false)
    std_false = round(std_false)

    # Display the results
    print("Mean Release Year (Month Available):", mean_true)
    print("Standard Deviation Release Year (Month Available):", std_true)
    print("\nMean Release Year (Month Not Available):", mean_false)
    print("Standard Deviation Release Year (Month Not Available):", std_false)
    # Create a list with the values
    data_counts = [nan_count, non_nan_count]

    # Create a list with the labels
    labels = ['Month release unavailable', 'Month release available']

    # Create the pie chart using Seaborn
    sns.set(style="whitegrid")  
    plt.figure(figsize=(6, 6))
    sns.set_palette("pastel")  
    plt.pie(data_counts, labels=labels, autopct='%1.1f%%', startangle=140)

    # Add a title 
    plt.title('Proportion of Month Release Availability')

    # Show the pie chart
    plt.show()

def plot_monthly_movie_counts(df, selected_genres = None):
    """
    Plot the number of movies released over the months for selected genres.

    Args:
        df (pd.DataFrame): DataFrame containing the movie data with a "Release Month" column.
        selected_genres (list): List of selected genres to filter the data.

    Returns:
        None
    """

    if selected_genres is None:
        # Count the number of movies released in each month and sort by month index
        month_counts = df["Release Month"].value_counts().sort_index()

        # Create a list of month names using the calendar library
        month_names = [calendar.month_name[i] for i in range(1, 13)]

        # Create a bar plot for the months
        plt.figure(figsize=(10, 6))
        sns.barplot(x=month_counts.index, y=month_counts.values)
        plt.title(f"Number of movies released over the months")
        plt.xlabel("Month")
        plt.ylabel("Number of movies")
        plt.xticks(range(12), month_names, rotation=45)  # You can adjust the rotation angle as per your preference
        plt.show()

    else:
        for genre in selected_genres:
            # Filter the DataFrame to include only rows where movies are available and belong to the selected genre
            df_filtered = filter_movies_by_genres(df, [genre])

            # Count the number of movies released in each month and sort by month index
            month_counts = df_filtered["Release Month"].value_counts().sort_index()

            # Create a list of month names using the calendar library
            month_names = [calendar.month_name[i] for i in range(1, 13)]

            # Create a bar plot for the months
            plt.figure(figsize=(10, 6))
            sns.barplot(x=month_counts.index, y=month_counts.values)
            plt.title(f"Number of {genre} movies released over the months")
            plt.xlabel("Month")
            plt.ylabel("Number of movies")
            plt.xticks(range(12), month_names, rotation=45)  # You can adjust the rotation angle as per your preference
            plt.show()

def plot_award_distribution(df, str):
    """
    Plot the pie chart for award dataset

    Args:
        df (pd.DataFrame): DataFrame containing the movie data with a "Release Month" column.
        str: string to control the output display

    Returns:
        None
    """

    country_counts = df['Movie '+str].value_counts()

    # Select the top 10 countries
    top_countries = country_counts[:5]

    # Sum up the rest of the countries
    other_countries_count = country_counts[5:].sum()

    # Add the 'Others' category to the top countries
    top_countries['Others'] = other_countries_count

    # Plotting the distribution
    ax = top_countries.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='tab20')

    # Adding labels and title
    plt.axis('equal')
    plt.title('Distribution of Movie '+str)
    ax.set_ylabel('')

    # Display the plot
    plt.show()