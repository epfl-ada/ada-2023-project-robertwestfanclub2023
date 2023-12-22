####################################################################################################
# IMPORTS
####################################################################################################
# Standard libraries
import calendar

# Third-party libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Plotly imports
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Statsmodels imports
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

from plotly.offline import plot

#####
# To save as html
#####

def to_html(fig, html_file_path):
    # Generate the HTML elements to embed the plot
    div = plot(fig, output_type='div', include_plotlyjs=True)

    # Create the full HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Interactive Plot</title>
    </head>
    <body>
        {div}
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(html_file_path, 'w') as html_file:
        html_file.write(html_content)
        
####################################################################################################
# Plot for Preprocessing Part
####################################################################################################*

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

####################################################################################################
# Research Questions 1
####################################################################################################


#########################################
# Exploratory functions
#########################################

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

def filter_dataframe_by_threshold(df, threshold=200):
    """
    Filter a DataFrame based on a specified threshold for the count ('Counts') per release year ('Release Year').

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        threshold (int): The threshold for the count below which years will be excluded.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only the rows where the count is greater than or equal to the threshold.
    
    """
    # Group the data by release year and count occurrences
    grouped = df.groupby(['Release Year']).size().reset_index(name='Counts')
    
    # Add a 'Keep' column to mark whether the count is above or equal to the threshold
    grouped['Keep'] = grouped['Counts'] >= threshold
    
    # Find the index of the last year below the threshold
    last_false_index = grouped[~grouped['Keep']]['Release Year'].values[-1]
    
    # Filter the original DataFrame, keeping only years above the threshold
    filtered_df = df[df['Release Year'] > int(last_false_index)]
    
    return filtered_df

def plot_histogram_by_release_year(df):
    """
    Plot a histogram of counts by release year.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        None
    """
    grouped = df.groupby(['Release Year']).size().reset_index(name='Counts')
    
    plt.figure(figsize=(10, 6))
    plt.bar(grouped['Release Year'], grouped['Counts'], color='darkblue')
    plt.xlabel("Release Year")
    plt.ylabel("Counts")
    plt.title('Histogram of Counts by Release Year')
    plt.show()
    

####################################################################
# Comprehensive Seasonality Analysis Across All Genres and Locations
####################################################################

def plot_monthly_average(df_mean_month):
    """
    Plot the monthly average of the percentage of movies released in a given month.

    Parameters:
    df_mean_month (DataFrame): A DataFrame containing the monthly mean percentage data.

    Returns:
    None
    """
    # Create a list of month names using the calendar library
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    # Create a bar plot for the months
    plt.figure(figsize=(10, 6))
    plt.bar(range(12), df_mean_month['Percentage'], color='darkblue')
    plt.xlabel("Release Month")
    plt.ylabel("Mean of the percentage of movies released in a given month (%)")
    plt.xticks(range(12), month_names, rotation=45)  
    plt.show()


def plot_acf_custom(data, max_lags=240, y_min=-0.2, y_max=1.0):
    """
    Generate and display an Autocorrelation Function (ACF) plot.

    Parameters:
    - data: The time series data for which ACF needs to be calculated and plotted.
    - max_lags: The maximum number of lags to consider (default is 240).
    - y_min: The minimum value for the y-axis (default is -0.2).
    - y_max: The maximum value for the y-axis (default is 1.0).

    Returns:
    None (displays the ACF plot).
    """

    # Creating the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting the ACF with dark blue color
    plot_acf(data, lags=max_lags, ax=ax, color="darkblue")

    # Custom x-axis formatting
    ticks = [12 * i for i in range(1, 21)]  # Multiples of 12 up to 240
    ticklabels = [f"{i} year{'s' if i > 1 else ''}" for i in range(1, 21)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=10, color="black")

    # Set the title, x-axis label, and y-axis label
    ax.set_title('Autocorrelation Function (ACF)')
    ax.set_xlabel('Lags (in months)')
    ax.set_ylabel('Autocorrelation')

    # Set the desired y-axis limits
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

def plot_seasonal_decomposition(df_past, df_recent):
    """
    Perform seasonal decomposition and plot the results for past and recent datasets.

    Args:
        df_past (pd.DataFrame): DataFrame containing past years' time series data.
        df_recent (pd.DataFrame): DataFrame containing recent years' time series data.

    Returns:
        None
    """
    # Perform multiplicative seasonal decomposition on the 'Counts' column with a period of 12 (monthly data) for the 'df_past' dataset
    result_past = seasonal_decompose(df_past['Counts'], model='multiplicative', period=12)

    # Perform multiplicative seasonal decomposition on the 'Counts' column with a period of 12 (monthly data) for the 'df_recent' dataset
    result_recent = seasonal_decompose(df_recent['Counts'], model='multiplicative', period=12)

    # Create subplots for 'df_past' and 'df_recent'
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    fig.suptitle("Seasonal Decomposition for Past and Recent Years", fontsize=16)

    # Plotting the decomposed components for 'df_past'
    colors_past = ['darkblue', 'darkblue', 'darkblue', 'darkblue']  
    axes[0, 0].plot(df_past['Counts'], color=colors_past[0], label='Original')
    axes[1, 0].plot(result_past.trend, color=colors_past[1], label='Trend')
    axes[2, 0].plot(result_past.seasonal, color=colors_past[2], label='Seasonal')
    axes[3, 0].plot(result_past.resid, color=colors_past[3], label='Residuals')

    axes[0, 0].set_title('Past Years Original Data', fontsize=14)
    axes[1, 0].set_title('Past Years Decomposition (Trend)', fontsize=14)
    axes[2, 0].set_title('Past Years Decomposition (Seasonal)', fontsize=14)
    axes[3, 0].set_title('Past Years Decomposition (Residuals)', fontsize=14)

    # Plotting the decomposed components for 'df_recent'
    colors_recent = ['darkblue', 'darkblue', 'darkblue', 'darkblue']
    axes[0, 1].plot(df_recent['Counts'], color=colors_recent[0], label='Original')
    axes[1, 1].plot(result_recent.trend, color=colors_recent[1], label='Trend')
    axes[2, 1].plot(result_recent.seasonal, color=colors_recent[2], label='Seasonal')
    axes[3, 1].plot(result_recent.resid, color=colors_recent[3], label='Residuals')

    axes[0, 1].set_title('Recent Years Original Data', fontsize=14)
    axes[1, 1].set_title('Recent Years Decomposition (Trend)', fontsize=14)
    axes[2, 1].set_title('Recent Years Decomposition (Seasonal)', fontsize=14)
    axes[3, 1].set_title('Recent Years Decomposition (Residuals)', fontsize=14)

    for ax in axes.ravel():
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.show()

    return result_past, result_recent

def plot_seasonal_components(result_past, result_recent):
    """
    Plots the seasonal components for 'Past' and 'Recent' datasets side by side.

    Args:
        result_past (seasonal decomposition result): Seasonal decomposition result for 'Past' dataset.
        result_recent (seasonal decomposition result): Seasonal decomposition result for 'Recent' dataset.

    Returns:
        None
    """
    # Extract the seasonal values for 'Past' dataset
    seasonal_values_past = result_past.seasonal[-12:]  # Replace 12 with the period of your data if different

    # Extract the seasonal values for 'Recent' dataset
    seasonal_values_recent = result_recent.seasonal[-12:]  # Replace 12 with the period of your data if different

    # Find the maximum value in the seasonal components
    max_seasonal_value = max(max(seasonal_values_past), max(seasonal_values_recent)) + 0.1

    # Create subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the 'Past' seasonal component
    axs[0].bar(x=np.arange(1, 13), height=seasonal_values_past, tick_label=[calendar.month_abbr[i] for i in range(1, 13)], color='darkblue')
    axs[0].set_title('Seasonal Component for Past Years [1976-1992]', fontsize=14)
    axs[0].set_xlabel('Month')
    axs[0].set_ylabel('Seasonal Effect')
    axs[0].set_ylim(0, max_seasonal_value)  # Set the same y-axis limits
    axs[0].grid(True)

    # Plot the 'Recent' seasonal component
    axs[1].bar(x=np.arange(1, 13), height=seasonal_values_recent, tick_label=[calendar.month_abbr[i] for i in range(1, 13)], color='darkblue')
    axs[1].set_title('Seasonal Component for Recent Years [1993-2009]', fontsize=14)
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('Seasonal Effect')
    axs[1].set_ylim(0, max_seasonal_value)  # Set the same y-axis limits
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


####################################################################
# Comprehensive Seasonality Analysis Across All Genres and Locations
####################################################################

def plot_genre_month_percentage(df_year):
    """
    Plots the percentage of movie releases by month for different genres.

    Args:
        df_year (DataFrame): The DataFrame containing movie data.

    Returns:
        None
    """
    # Suppress user warnings
    warnings.filterwarnings("ignore")

    # Find unique movie genres in the DataFrame
    genres = df_year['Movie genres'].unique()

    # Create a figure with appropriate size for all subplots
    plt.figure(figsize=(15, 5 * len(genres)))

    # Loop through each genre to create a bar plot
    for i, genre in enumerate(genres):
        # Create a subplot for each genre
        plt.subplot(len(genres), 1, i + 1)

        # Filter the DataFrame for the current genre
        df_genre = df_year[df_year['Movie genres'] == genre]

        # Create the bar plot using seaborn
        sns.barplot(data=df_genre, x='Release Month', y='Percentage', hue='Movie Continent', errorbar=None)

        # Set the x-axis labels to month abbreviations
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        plt.xticks(range(12), month_names)

        # Add a title to the subplot
        plt.title(f'Percentage by Month for {genre}')

        # Add the legend in the upper right corner
        plt.legend(loc='upper right')

        # Adjust subplots automatically to fit within the figure
        plt.tight_layout()

    # Show the plots
    plt.show()

def calculate_correlations(df, number_years=4):
    """
    Calculate correlations for a given number of years using the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        number_years (int): The number of years for which you want to calculate correlations.

    Returns:
        pd.DataFrame: A DataFrame containing the correlations.
    """
    # Initialize an empty DataFrame to store correlations
    correlations = pd.DataFrame()

    for continent in df['Movie Continent'].unique():
        for genre in df['Movie genres'].unique():
            data = df[(df['Movie genres'] == genre) & (df['Movie Continent'] == continent)]

            for lag in range(1, number_years + 1):
                acf_vals = acf(data['Counts'], nlags=lag * 12)
                acf_df = pd.DataFrame({
                    'Movie genres': genre,
                    'Movie Continent': continent,
                    'Lag': f'{lag} year(s)',
                    'Correlation': [acf_vals[lag * 12]]
                })

                correlations = pd.concat([correlations, acf_df], axis=0)
    
    return correlations

def plot_correlation_heatmap(df, number_years=4):
    """
    Plot a correlation heatmap for a given number of years using the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        number_years (int): The number of years for which you want to calculate correlations.

    Returns:
        None (displays the heatmap).
    """
    # Initialize an empty DataFrame to store correlations
    correlations = calculate_correlations(df, number_years)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    pivot_df = correlations.pivot(values='Correlation', index=['Movie genres', 'Movie Continent'], columns='Lag')
    
    # Sort the columns chronologically
    sorted_columns = [f'{i} year(s)' for i in range(1, number_years + 1)]
    pivot_df = pivot_df[sorted_columns]
    
    sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap ({number_years} Year Lag)')
    plt.show()

def plot_seasonality_heatmap(decomposition_results):
    """
    Plot a heatmap for seasonality factors of movies across genres and continents.

    Parameters:
    decomposition_results (DataFrame): A DataFrame containing the decomposed components of seasonality
                                      with 'Movie genres', 'Movie Continent', and 'Release Month' as columns.

    """
    # Pivot the data to create a matrix suitable for a heatmap
    heatmap_data = decomposition_results.pivot_table(values='Seasonality', index=['Movie genres', 'Movie Continent'], columns='Release Month')
    heatmap_data.columns = [calendar.month_abbr[i] for i in range(1, 13)]
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", vmin=0, vmax=2.2)
    plt.title('Seasonality Factors Heatmap')
    plt.ylabel('Genre - Continent')
    plt.xlabel('Release Month')
    plt.show()

####################################################################################################
# Research Questions 2
####################################################################################################

def plot_box_office_oscars(df):
    """
    Plot a box plot to visualize the distribution of box office revenue for Oscar winners and non-winners.

    Parameters:
        df (DataFrame): A pandas DataFrame containing movie data, including the 'Winner Binary' column (1 for winners,
                        0 for non-winners) and 'Movie box office revenue'.

    """
    df_plot = df[['Winner Binary', 'Movie box office revenue']].copy()

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x='Winner Binary', y='Movie box office revenue', data=df_plot, palette=['blue', 'red'])

    plt.title('Distribution of Box Office Revenue for Oscar Winners and Non-Winners')
    plt.xlabel('Oscar Winner (1: Yes, 0: No)')
    plt.ylabel('Box Office Revenue')
    plt.yscale('log')  # Set y-axis to log scale
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.show()

def plot_ratings_oscars(df):
    """
    Plot a box plot to visualize the distribution of average ratings for Oscar winners and non-winners.

    Parameters:
        df (DataFrame): A pandas DataFrame containing movie data, including the 'Winner' column (1 for winners,
                        0 for non-winners) and 'Average Vote' for average ratings.

    """
    df_plot = df[['Winner', 'Average Vote ']].copy()

    df_plot['Winner'] = df_plot['Winner'].astype(int)
    df_plot['Average Vote '] = df_plot['Average Vote '].astype(float)

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x='Winner', y='Average Vote ', data=df_plot, palette=['blue', 'red'])

    plt.title('Distribution of Average Ratings for Oscar Winners and Non-Winners')
    plt.xlabel('Oscar Winner (1: Yes, 0: No)')
    plt.ylabel('Average Ratings')
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.show()

def plot_column_by_oscars_category(df, column, yscale=False):
    """
    Plots a box plot of the specified column for Oscar-winning movies in each category.

    Parameters:
        df (DataFrame): The input DataFrame containing movie data.
        column (str): The name of the column to plot.
        yscale (bool, optional): Whether to use a log scale for the y-axis (default is False).

    Returns:
        None

     """
    # Filter the DataFrame for movies that have won Oscars and have non-NaN values in the specified column
    df_oscar_winners = df[(df['Winner'] == 1) & (~df[column].isna())]

    # Filter out categories with fewer than a certain threshold of movies
    category_threshold = 5  # Adjust as needed
    category_counts = df_oscar_winners['Category'].value_counts()
    valid_categories = category_counts[category_counts >= category_threshold].index
    df_oscar_winners_filtered = df_oscar_winners[df_oscar_winners['Category'].isin(valid_categories)]

    # Calculate the mean of the specified column for each category and sort by mean in ascending order
    mean_column_by_category = df_oscar_winners_filtered.groupby('Category')[column].mean().sort_values(ascending=True)

    # Set up the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    if yscale:
        plt.yscale('log')  # Set y-axis to log scale for better visualization

    # Create the box plot for each category, sorted by mean column values
    sns.boxplot(x='Category', y=column, data=df_oscar_winners_filtered,
                order=mean_column_by_category.index, palette='Set1')

    plt.title(f'Distribution of {column} for Oscar Winners in Each Category (Filtered)')
    plt.xlabel('Oscar Category')
    if yscale:
        plt.ylabel(f'{column} (log scale)')
    else:
        plt.ylabel(column)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

####################################################################################################
# Research Questions 3
####################################################################################################

def plot_monthly_votes_and_ratings(df):
    """
    Plot histograms of average vote and rating by month using Plotly.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        None (displays the plot).
    """

    # Group by month and calculate average vote average and rating
    monthly_votes = df.groupby('Release Month')['Average Vote '].mean()
    monthly_ratings = df.dropna(subset=['Rating']).groupby('Release Month')["Rating"].mean()

    # Create subplots with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Vote average on the left y-axis
    fig.add_trace(
        go.Bar(x=monthly_votes.index, y=monthly_votes.values, name='Vote Avg', marker_color='blue'),
        secondary_y=False
    )

    # Rating on the right y-axis
    fig.add_trace(
        go.Bar(x=monthly_ratings.index, y=monthly_ratings.values, name='Rating', marker_color='red'),
        secondary_y=True
    )

    # Update x-axis and y-axes labels
    fig.update_xaxes(title_text='Month', tickvals=list(range(1, 13)),
                     ticktext=[calendar.month_name[i] for i in range(1, 13)])
    fig.update_yaxes(title_text='Vote Average', secondary_y=False)
    fig.update_yaxes(title_text='Rating', secondary_y=True)

    # Add layout details
    fig.update_layout(
        title='Average Vote Average and Rating by Month',
        barmode='group'
    )

    # Show the plot
    fig.show()

def plot_adjusted_box_office_revenue(merged_df):
    """
    Plot a bar chart of the average adjusted box office revenue by month using Plotly.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None (displays the plot).
    """

    # Create a bar plot using Plotly Express
    fig = px.bar(
        merged_df, 
        x='Release Month', 
        y='Ratio Revenue', 
        labels={'Ratio Revenue': 'Average Adjusted Revenue'},
        title='Average Adjusted Box Office Revenue by Month'
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Average Adjusted Revenue'
    )

    # Show the plot
    fig.show()

def plot_movie_metrics_evolution(yearly_data):
    """
    Plot a line chart showing the evolution of movie metrics over years using Plotly.

    Args:
        yearly_data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None (displays the plot).
    """

    # Create a line plot using Plotly Express
    fig = px.line(
        yearly_data, 
        title='Evolution of Movie Metrics Over Years',
        labels={'value': 'Average Values', 'variable': 'Metrics'}
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Values'
    )

    # Show the plot
    fig.show()

def plot_seasonal_votes_and_ratings(df):
    """
    Plot histograms of average vote average and rating by season using Plotly.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None (displays the plot).
    """

    # Group by season and calculate average vote average and rating
    seasonal_votes = df.groupby('season')['Average Vote '].mean()
    seasonal_ratings = df.groupby('season')['Rating'].mean()

    # Create subplots with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Vote average on the left y-axis
    fig.add_trace(
        go.Bar(x=seasonal_votes.index, y=seasonal_votes.values, name='Vote Avg', marker_color='blue'),
        secondary_y=False
    )

    # Rating on the right y-axis
    fig.add_trace(
        go.Bar(x=seasonal_ratings.index, y=seasonal_ratings.values, name='Rating', marker_color='red'),
        secondary_y=True
    )

    # Update x-axis and y-axes labels
    fig.update_xaxes(title_text='Season')
    fig.update_yaxes(title_text='Vote Average', secondary_y=False)
    fig.update_yaxes(title_text='Rating', secondary_y=True)

    # Add layout details
    fig.update_layout(
        title='Average Vote Average and Rating by Season',
        barmode='group'
    )

    # Show the plot
    fig.show()

def plot_rating_and_vote_vs_box_office(df):
    """
    Plot scatter plots for 'Rating' vs 'Box Office Revenue' and 'Vote Average' vs 'Box Office Revenue' using Plotly.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None (displays the plots).
    """

    # Calculate correlation coefficients
    corr_rating = df['Rating'].corr(df['Movie box office revenue'])
    corr_vote_average = df['Average Vote '].corr(df['Movie box office revenue'])

    # Print the correlation coefficients
    print("Correlation between Rating and Box Office Revenue:", corr_rating)
    print("Correlation between Vote Average and Box Office Revenue:", corr_vote_average)

    # Scatter plot for 'Rating' vs 'Movie box office revenue'
    fig_rating = px.scatter(
        df, 
        x='Rating', 
        y='Movie box office revenue', 
        title='Rating vs Box Office Revenue',
        labels={'Rating': 'Rating', 'Movie box office revenue': 'Box Office Revenue'}
    )
    fig_rating.show()

    # Scatter plot for 'Vote Average' vs 'Movie box office revenue'
    fig_vote = px.scatter(
        df, 
        x='Average Vote ', 
        y='Movie box office revenue', 
        title='Vote Average vs Box Office Revenue',
        labels={'Average Vote ': 'Vote Average', 'Movie box office revenue': 'Box Office Revenue'}
    )
    fig_vote.show()
