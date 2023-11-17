# Cinematic Dynamics: Exploring Film Genres, Release Timing, and Success Factors

## Abstract:

Our project explores the dynamic landscape of film releases, investigating their adherence to seasonal trends and awards ceremonies. Our curiosity extends to understanding how various genres gain popularity across different regions and throughout the year. For example, the rarity of Christmas movies being released outside the holiday season and the strategic release of award-worthy films during periods like the Oscars intrigue us. By deciphering these patterns, we aim to predict future genre trends and examine how the timing of a movie's release as well as the actor's choice influences its success, based on several factors. Alongside movie and character metadata from Freebase, we use several additional datasets to encompass the effect of release period and actor choice on the popularity of a movie based on its box office revenue, user ratings, and awards received.

## Datasets

The main dataset comprises metadata extracted from Freebase, originally used in the CMU Movie Summary Corpus dataset, which will not be utilized in our analysis. This dataset encompasses information on 81,741 films and was extracted from the Freebase database dated November 4, 2012. Covering the period from 1915 to 2012, it includes a diverse range of historical films. The metadata provides details about movies, such as box office revenue, genre, release period, language, and aligned information about characters and the actors portraying them (name, gender, age, etc.). Additional information about this dataset can be found on the CMU Movie Summary Corpus page: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/), and in the dataset README, which includes column names: [README.txt]( https://www.cs.cmu.edu/~ark/personas/data/README.txt).

To enhance our dataset and address missing values, we incorporated several additional datasets, including information on ratings and awards:

- MovieLens: [This MovieLens dataset](https://files.grouplens.org/datasets/movielens/ml-25m-README.html) is a widely used dataset in the field of recommender systems, containing millions of movie ratings and tag assignments collected from the MovieLens website. It offers detailed information about movies, users, and their interactions, making it an essential resource for research and development in personalized recommendation algorithms. We will use it due to its extensive collection of movie ratings, which provides valuable insights for analyzing viewer preferences and movie popularity trends (The notebook additional_dataset.ipynb contains both the code and the explanations detailing how we preprocessed this dataset to construct our own tailored additional dataset).
- The Movie Dataset : Our analysis heavily relies on the exact release dates of movies. However, 50% of our dataset only includes the year of release, not the month. Therefore, we will endeavor to extract as many exact release dates as possible from [this Kaggle dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data). Additionally, this dataset contains IMDb votes and ratings, which could also prove useful for our analysis.
- IMDb: This dataset provides average ratings and the number of votes for more than three million movies, accessible at: [IMDb Dataset](https://developer.imdb.com/non-commercial-datasets/).
- Academy Awards Database: A comprehensive record of Oscar nominees and winners, encompassing movie names, year, and category, since the first Oscars to the present day. This dataset can be found at: [Academy Awards Database](https://www.oscars.org/oscars/awards-databases).

## Research Questions:

A list of research questions you would like to address during the project.

- Question 1: Are there any recurring patterns between a film's genre and its release timing within a year? If such patterns exist, do they vary based on the location of the movie, and can these insights contribute to predicting the genre of upcoming film releases in subsequent seasons?
- Question 2: Which factors have an impact on a movie's success, particularly its likelihood of winning an Oscar award? How does the effect of release timing evolve over time? Is it possible to predict the Oscar probability, given some factors such as release month, country, language, etc.?
- Question 3: How does the release month of a movie influence its overall success and popularity, as evidenced by box office revenues, the number of votes, and ratings, while also accounting for the evolution of data documentation quality over the years?
- Question 4: To what extent does the involvement of specific actors influence a film's success, including its chances of winning awards, expected box office revenue, and ratings? Additionally, can we generate actor groups based on genres and predict their probability of achieving success?

## Methods

- Descriptive Analysis: Conducting an exhaustive examination of the historical evolution of genre popularity and seasonal trends. This involves dissecting patterns in film releases, identifying genre shifts over time, and understanding their correlation with seasonal fluctuations.

- Causal Discovery: Delving into the causal factors influencing changes in genre popularity. Through a systematic analysis, we aim to identify and understand the key drivers that contribute to shifts in the popularity of different film genres.

- Exploratory Analysis: An in-depth examination of the correlation between release timing, box office success, and potential awards. This involves dissecting the relationships between the timing of film releases, their financial success, and the recognition they receive in prestigious awards ceremonies.

- Predictive Analysis: Investigating the impact of specific actors on a film's success, including their influence on winning awards, expected box office revenue, and ratings. We also aim to generate actor groups based on genres and predict their probability of achieving success; Predicting the probability that one movie can win the Oscar Award given key factors including actors, movie language, country, and release month.

## Proposed Timeline:

- 17.11.2003: Finish milestone P2, start Homework 2
- 24.11.2023: Complete project directions given milestone P2 feedback and initial analyses
- 01.12.2023: Homework 2 deadline, begin final analysis
- 08.12.2023: Draft data story and finish analysis
- 15.12.2023: Finish data story, start website and final review
- 23.12.2023: Milestone P3 deadline

## Organisation within the team:

- Léa: Movie metadata cleaning, RQ1
- Salim: Additional dataset : MovieLens dataset and Kaggle's dataset processing and merging
- Jason: include IMDb ratings, RQ3
- Pierre-Hadrien: Utilize Character metadata, RQ4
- Yanruiqi: Process and analyze on Awards Dataset, RQ2
