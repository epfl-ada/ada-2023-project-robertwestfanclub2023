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
    
    # Create a histogram of counts by release year
    fig = px.bar(grouped, x='Release Year', y='Counts', orientation='v', 
                 title='Histogram of Counts by Release Year')
    fig.update_xaxes(title_text='Release Year')
    fig.update_yaxes(title_text='Counts')
    fig.show()
    
    return filtered_df

####################################################################
# Comprehensive Seasonality Analysis Across All Genres and Locations
####################################################################

def plot_monthly_average(df_mean_month, color='#636EFA', html_file_path_name = "images/1_mean_percent_release_month_histo.html"):
    """
    Plot a bar chart showing the monthly average percentage of movies released using Plotly.

    Args:
        df_mean_month (pd.DataFrame): DataFrame containing the mean percentage of movies released in each month.
        color (str): Hex color code for the bars in the plot.

    Returns:
        None
    """

    # Create a list of month names using the calendar library
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    # Create a DataFrame for Plotly
    plot_data = pd.DataFrame({
        'Month': month_names,
        'Mean Percentage': df_mean_month['Percentage']
    })

    # Create a bar plot for the months using Plotly
    fig = px.bar(plot_data, x='Month', y='Mean Percentage', title='Monthly Average Percentage of Movies Released',
                 color_discrete_sequence=[color])

    # Customize the layout
    fig.update_layout(xaxis_title='Release Month', yaxis_title='Mean Percentage (%)')
    
    to_html(fig, html_file_path_name)

    # Show the plot
    fig.show()

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

def plot_seasonal_decomposition(df_past, df_recent, color='darkblue', html_file_path_name = "images/1_season_decomposition.html"):
    """
    Perform seasonal decomposition and plot the results for past and recent datasets using Plotly.

    Args:
        df_past (pd.DataFrame): DataFrame containing past years' time series data.
        df_recent (pd.DataFrame): DataFrame containing recent years' time series data.
        color (str): Color for the plot lines.

    Returns:
        Tuple: Decomposition results for past and recent data.
    """
    # Seasonal decomposition
    result_past = seasonal_decompose(df_past['Counts'], model='multiplicative', period=12)
    result_recent = seasonal_decompose(df_recent['Counts'], model='multiplicative', period=12)

    # Create subplots
    fig = make_subplots(rows=4, cols=2, subplot_titles=("Past Years Original", "Recent Years Original",
                                                        "Past Years Trend", "Recent Years Trend",
                                                        "Past Years Seasonal", "Recent Years Seasonal",
                                                        "Past Years Residuals", "Recent Years Residuals"))

    # Plotting for 'df_past'
    fig.add_trace(go.Scatter(x=df_past.index, y=df_past['Counts'], mode='lines', name='Original', line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_past.index, y=result_past.trend, mode='lines', name='Trend', line=dict(color=color)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_past.index, y=result_past.seasonal, mode='lines', name='Seasonal', line=dict(color=color)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_past.index, y=result_past.resid, mode='lines', name='Residuals', line=dict(color=color)), row=4, col=1)

    # Plotting for 'df_recent'
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Counts'], mode='lines', name='Original', line=dict(color=color)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_recent.index, y=result_recent.trend, mode='lines', name='Trend', line=dict(color=color)), row=2, col=2)
    fig.add_trace(go.Scatter(x=df_recent.index, y=result_recent.seasonal, mode='lines', name='Seasonal', line=dict(color=color)), row=3, col=2)
    fig.add_trace(go.Scatter(x=df_recent.index, y=result_recent.resid, mode='lines', name='Residuals', line=dict(color=color)), row=4, col=2)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text="Seasonal Decomposition for Past and Recent Years", showlegend=False)
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=2)
    fig.update_yaxes(title_text="Value", col=1)
    fig.update_yaxes(title_text="Value", col=2)

    to_html(fig, html_file_path_name)

    # Show the plot
    fig.show()

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
    Plots the percentage of movie releases by month for different genres in a Plotly format

    Args:
        df_year (DataFrame): The DataFrame containing movie data.

    Returns:
        A Plotly figure object.
    """
    # Define a color map for continents
    color_map = {
        'Asia': 'blue',
        'Europe': 'red',
        'North America': 'green'
    }

    # Find unique movie genres in the DataFrame
    genres = df_year['Movie genres'].unique()

    # Create a subplot figure with one row for each genre
    fig = make_subplots(rows=len(genres), cols=1, subplot_titles=[f'Percentage by Month for {genre}' for genre in genres])

    first_genre = True

    # Loop through each genre to create a bar plot
    for i, genre in enumerate(genres):
        # Filter the DataFrame for the current genre
        df_genre = df_year[df_year['Movie genres'] == genre]

        # Create a bar plot for each continent within the current genre
        for continent in df_genre['Movie Continent'].unique():
            df_continent = df_genre[df_genre['Movie Continent'] == continent]
            # Group by 'Release Month' and calculate the mean percentage
            df_grouped = df_continent.groupby('Release Month')['Percentage'].mean().reset_index()

            # Add trace to the subplot for the current genre
            fig.add_trace(
                go.Bar(
                    x=[calendar.month_abbr[month] for month in df_grouped['Release Month']],
                    y=df_grouped['Percentage'],
                    name=continent,
                    marker_color=color_map[continent],  # Use the color map for consistent colors
                    showlegend=first_genre  # Show legend only for the first genre
                ),
                row=i+1,
                col=1
            )

        # After the first genre, set this to False so that legends are not duplicated
        first_genre = False

    # Update layout for a cleaner look
    fig.update_layout(
        height=300 * len(genres),
        title_text="Movie Release Distribution by Genre and Month",
        barmode='group'
    )

    # Adjust x-axis and y-axis titles
    for i in range(len(genres)):
        fig['layout'][f'xaxis{i+1}'].update(title='Release Month')
        fig['layout'][f'yaxis{i+1}'].update(title='Percentage')

    # Set legend position
    fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.95, xanchor="right", x=1.3))

    return fig

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
