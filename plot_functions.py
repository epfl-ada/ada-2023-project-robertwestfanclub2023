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
    Plot a pie chart showing the proportion of movies with available and unavailable release months in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing movie data.

    Returns:
        None
    """

     # Calculate the number of NaN and non-NaN values
    non_available = df[df['Release Month'] == -1].shape[0]
    available = df[df['Release Month'] != -1].shape[0]

    # Print the number of movies with missing release month and the total number of movies in the dataset
    print("In this dataset, there are {} movies with missing release month out of a total of {} movies.".format(non_available, non_available + available))

    # Create a list with the values
    data_counts = [non_available, available]

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

def plot_movie_continent_distribution(df):
    """
    Plot a pie chart to visualize the distribution of movies by continent in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing movie data.

    Returns:
        None
    """
    # Extract labels and data counts for the movie continents
    labels = df['Movie Continent'].value_counts().index.tolist()
    data_counts = df['Movie Continent'].value_counts().values.tolist()

    # Set the style for the plot
    sns.set(style="whitegrid")
    
    # Create a figure with a specified size and pastel color palette
    plt.figure(figsize=(8, 6))
    sns.set_palette("pastel")

    # Create a pie chart without exploded segments and set the starting angle
    plt.pie(data_counts, startangle=140)

    # Set the aspect to 'equal' for a circular pie chart
    plt.gca().set_aspect('equal')

    # Add a title to the chart
    plt.title("Movie Continent Distribution")

    # Calculate percentages for the legend
    total = sum(data_counts)
    percentages = [f"{count / total * 100:.1f}%" for count in data_counts]

    # Create a legend with labels and percentages, placed to the side of the chart
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        labels=[f"{label} ({percentage})" for label, percentage in zip(labels, percentages)]
    )

    # Show the plot
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

    # Explode the top 2 countries
    explode = [0.1 if i < 2 else 0 for i in range(len(top_countries))]

    # Plotting the distribution
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts = ax.pie(top_countries, startangle=90, explode=explode, labels=top_countries.index, pctdistance=0.8, textprops=dict(color="w"))

    # Adding labels and title
    plt.axis('equal')
    plt.title('Distribution of Movie '+str)

    # Creating a legend with percentage labels
    legend_labels = [f"{label} ({value} movies, {percentage:.1f}%)"
                     for label, value, percentage in zip(top_countries.index, top_countries.values, top_countries.values / top_countries.values.sum() * 100)]

    plt.legend(wedges, legend_labels, title=str.capitalize(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Display the plot
    plt.show()

def plot_box_office_oscars(df):

    df_plot = df[['Winner', 'Movie box office revenue']].copy() 

    # Convert 'Winner' column to numeric (True/False to 1/0)
    df_plot['Winner'] = df_plot['Winner'].astype(int)

    # Plot using seaborn

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x='Winner', y='Movie box office revenue', data=df_plot, palette={0: 'blue', 1: 'red'})

    plt.title('Distribution of Box Office Revenue for Oscar Winners and Non-Winners')
    plt.xlabel('Oscar Winner (1: Yes, 0: No)')
    plt.ylabel('Box Office Revenue')
    plt.yscale('log')  # Set y-axis to log scale
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.show()

def plot_ratings_oscars(df):

    df_plot = df[['Winner', 'Average Vote ']].copy() 

    df_plot['Winner'] = df_plot['Winner'].astype(int)
    df_plot['Average Vote '] = df_plot['Average Vote '].astype(float)

    # Plot using seaborn
    
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x='Winner', y='Average Vote ', data=df_plot, palette={0: 'blue', 1: 'red'})

    plt.title('Distribution of Average Ratings for Oscar Winners and Non-Winners')
    plt.xlabel('Oscar Winner (1: Yes, 0: No)')
    plt.ylabel('Average Ratings')
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.show()

def plot_total_box_office(actor_revenue):
    
    # Plot the distribution
    plt.figure(figsize=(12, 6))
    actor_revenue.plot(kind='bar', color='skyblue')
    
    plt.title('Total Box Office Revenue by Actor (Top 50)')
    plt.xlabel('Actor')
    plt.ylabel('Total Box Office Revenue')
    
    plt.show()


def plot_column_by_oscars_category(df, column,yscale):
    """
    Plots a box plot of the box office revenue for Oscar-winning movies in each category.

    Parameters:
    df (DataFrame): The input DataFrame containing movie data.
    column (String): the column to plot
    yscale (Boolean): whether to use log scale for y-axis
    Returns:
    None
    """

    # Filter the DataFrame for movies that have won Oscars and have non-NaN box office revenue
    df_oscar_winners = df[(df['Winner'] == 1) & (~df[column].isna())]

    # Filter out categories with fewer than a certain threshold of movies
    category_threshold = 5  # Adjust as needed
    category_counts = df_oscar_winners['Category'].value_counts()
    valid_categories = category_counts[category_counts >= category_threshold].index
    df_oscar_winners_filtered = df_oscar_winners[df_oscar_winners['Category'].isin(valid_categories)]

    # Calculate mean box office revenue for each category and sort by mean in ascending order
    mean_revenue_by_category = df_oscar_winners_filtered.groupby('Category')[column].mean().sort_values(ascending=True)

    # Set up the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    if yscale:
        plt.yscale('log')  # Set y-axis to log scale for better visualization

    # Create the box plot for each category, sorted by mean box office revenue
    sns.boxplot(x='Category', y=column, data=df_oscar_winners_filtered,
                order=mean_revenue_by_category.index, palette='Set1')

    plt.title(f'{column} for Oscar Winners in Each Category (Filtered)')
    plt.xlabel('Oscar Category')
    plt.ylabel(column)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()
