# Cinematic Dynamics: Exploring Film Genres, Release Timing, and Success Factors

Datastory
Discover how the seasons shape the cinematic landscape by diving into our fascinating datastory. Follow this link to explore the captivating impact of time on the world of cinema : [Is Timing Everything ?](https://scherkao31.github.io/timing_is_everything/)

## Abstract:

Our project delves into the dynamic landscape of film releases, investigating their adherence to seasonal trends, awards ceremonies, and the broader factors that contribute to their success. Our curiosity extends to understanding how various genres gain popularity across different regions and throughout the year. For example, the rarity of Christmas movies being released outside the holiday season and the strategic release of award-worthy films during periods like the Oscars intrigue us. By deciphering seasonal, locational, and success-based trends, we aim to examine how those factors influence its success, based on other factors such as awards, ratings, and box office revenue. Alongside movie and character metadata from Freebase, we use several additional datasets to encompass all necessary factors. This exploration will prove interesting for movie enthusiasts, as well as for film producers and distributors, who can leverage our findings to optimize their release strategies.

## Datasets

The main dataset comprises metadata extracted from Freebase, originally used in the CMU Movie Summary Corpus dataset, which will not be utilized in our analysis. This dataset encompasses information on 81,741 films and was extracted from the Freebase database dated November 4, 2012. Covering the period from 1915 to 2012, it includes a diverse range of historical films. The metadata provides details about movies, such as box office revenue, genre, release period, language, and aligned information about characters and the actors portraying them (name, gender, age, etc.). Additional information about this dataset can be found on the CMU Movie Summary Corpus page: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/), and in the dataset README, which includes column names: [README.txt](https://www.cs.cmu.edu/~ark/personas/data/README.txt).

To enhance our dataset and address missing values, we incorporated several additional datasets, including information on ratings and awards:

- MovieLens: [This MovieLens dataset](https://files.grouplens.org/datasets/movielens/ml-25m-README.html) is a widely used dataset in the field of recommender systems, containing millions of movie ratings and tag assignments collected from the MovieLens website. It offers detailed information about movies, users, and their interactions, making it an essential resource for research and development in personalized recommendation algorithms. We will use it due to its extensive collection of movie ratings, which provides valuable insights for analyzing viewer preferences and movie popularity trends (The notebook additional_dataset.ipynb contains both the code and the explanations detailing how we preprocessed this dataset to construct our own tailored additional dataset).
- The Movie Dataset: Our analysis heavily relies on the exact release dates of movies. However, 50% of our dataset only includes the year of release, not the month. Therefore, we will endeavor to extract as many exact release dates as possible from [this Kaggle dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data). Additionally, this dataset contains IMDb votes and ratings, which could also prove useful for our analysis.
- IMDb: This dataset provides average ratings and the number of votes for more than three million movies, accessible at [IMDb Dataset](https://developer.imdb.com/non-commercial-datasets/).
- Academy Awards Database: A comprehensive record of Oscar nominees and winners, encompassing movie names, years, and category, since the first Oscars to the present day. This dataset can be found at: [Academy Awards Database](https://www.oscars.org/oscars/awards-databases).

## Research Questions:

A list of research questions you would like to address during the project.

- Question 1: Are there any recurring patterns between a film's genre and its release timing within a year? If such patterns exist, do they vary based on the location of the movie, and can these insights contribute to predicting the genre of upcoming film releases in subsequent seasons?
- Question 2: How does the release timing of a movie impact its likelihood of winning an Oscar award? Which other factors have an impact on these chances?
- Question 3: How does the release month of a movie influence its overall success and popularity, as evidenced by box office revenues, the number of votes, and ratings, while also accounting for the evolution of data documentation quality over the years?

This question was considered too broad to include in our analysis, so we decided not to include it after milestone 2:

- Question 4: To what extent does the involvement of specific actors influence a film's success, including its chances of winning awards, expected box office revenue, and ratings? Additionally, can we generate actor groups based on genres and predict their probability of achieving success?

## Methods

### Research Question 1:

**Exploration and Data Preprocessing**:
   - Calculation of monthly averages of film releases across all genres and locations.
   - Use of bar charts to understand the distribution of film releases throughout the year.

**Comprehensive Seasonality Analysis**:
   - Examination of seasonality in film releases without genre or location bias.
   - Utilization of the Canova-Hansen Test to assess the stability of seasonal patterns.
   - Autocorrelation analysis to understand the persistence of seasonal patterns over time.

**Detailed Study by Genre and Location**:
   - Analysis of the most recent decade to study seasonal patterns in different film genres and locations.
   - Histogram analysis and Canova-Hansen Testing to understand the dynamics of cinematic releases by month, genre, and location.
   - Heatmap visualization to identify and compare seasonal patterns across different genres and regions.

### Research Question 2:  

**Exploratory Analysis:**  
An in-depth examination of the correlation between release timing, box office success, and potential awards. This involves dissecting the relationships between the timing of film releases, their financial success, and the recognition they receive in prestigious awards ceremonies.

**Confounder Analysis:**  
We take the bias from locations and release years (movie eras) into account and eliminate their effects by control variable (location) and normalization (release year).

**Clustering:**  
We clustered the percentage of winning movies for the release months to explore the similarity in their influences on the Oscar outcome.

**Statistical Validation:**  
- T-test: we use Welch's t-test to compare the means of two groups of data, in order to determine whether there is a significant difference between them. In our case, we use it to compare the box office revenue or average ratings between movies.

- Person's correlation: we use Person's correlation to measure the strength of a linear relationship between two variables. In our case, we use it to measure the correlation between the box office revenue or average ratings with the probability of winning an Oscars, amongst other analysis.

- Cramér's V: we use Cramér's V to test the correlation between release timing (months) and the binary variable of winning the Oscar. 

### Research Question 3:  

Linked to question 4 (not included in our final analysis, see milestone 2 feedback):

**Exploration and Data Preprocessing:**
- Dataset: Utilized the CBU movie dataset focusing on release dates, box office revenues, ratings, and genres. Utilized the MovieLens dataset for better ratings.
- Normalization: Normalized box office revenues against yearly averages for fair comparison over time.

**Analytical Approach:**
- Ratings and Revenue Correlation: Examined the relationship between movie ratings and box office revenue.
- Monthly and Genre-Based Analysis: Analyzed box office performance across different months and genres using visual tools like bar graphs and heatmaps.

**Seasonal Analysis:**
- Seasonal Trends: Investigated how ratings and box office revenues vary by season and genre.
- 'Blockbuster Seasons' and 'Dump Months': Explored seasonal impacts on movie release strategies.

**Statistical Techniques:**
Propensity Score Estimation: Used logistic regression and nearest-neighbor matching to analyze the impact of release month on movie popularity, indicated by votes and ratings.

## Timeline:

- 17.11.2003: Finished milestone P2
- 24.11.2023: Completed project directions given milestone P2 feedback and initial analyses
- 01.12.2023: Began final analysis
- 08.12.2023: Drafted data story and finished analysis
- 15.12.2023: Started website 
- 23.12.2023: Final review before Milestone P3 deadline

## Organisation within the team:

Milestone 2:

- Léa: Movie metadata cleaning, RQ1
- Salim: Additional dataset processing and merging (MovieLens dataset and Kaggle's dataset)
- Yanruiqi: Processing and analysis on Awards Dataset, RQ2
- Jason: include IMDb ratings, RQ3
- Pierre-Hadrien: Utilize Character metadata, RQ4

Milestone 3:

- Léa: Data preprocessing and analysis of research question 1
- Yanruiqi: Analysis of research question 2 (seasonal and locational influences on winning Oscar) 
- Pierre-Hadrien: README maintenance and analysis of research question 2 (impact of movies' success and other factors on their chances of winning an Oscar)
- Jason: Analysis of research question 3
- Salim: Preprocessing of the Movie Lens dataset, analysis of research question 3, and data story webpage deployment
