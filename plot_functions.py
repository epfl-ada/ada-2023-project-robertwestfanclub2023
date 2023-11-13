import seaborn as sns
import matplotlib.pyplot as plt
import calendar

import preprocessing
from preprocessing import filter_movies_by_genres

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
