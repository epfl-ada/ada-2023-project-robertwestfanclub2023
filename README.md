# Investigation on patent personas of Film characters

## Abstract:

Our project explores how film genres change over time and follow seasonal trends in releases. We're curious about how different genres become popular in various places and throughout the year. For instance, it's rare to see Christmas movies being released in other parts of the year, and directors eager to win an award often release movies close to the Oscar's or Festival de Cannes period. By understanding these patterns, we want to predict future genre trends and see how release timing affects a movie's success at the box office. Alongside the main dataset, CMU Movie Summary Corpus, which has a lot of movie details, we plan to include information about awards. This extra data will help us see how release timing based on location affects winning awards. Using descriptive analysis, causal discovery, and exploratory analysis, our project aims to shed light on the fascinating dynamics of the film industry.

## Datasets

The main dataset comprises metadata extracted from Freebase, originally used in the CMU Movie Summary Corpus dataset, which will not be utilized in our analysis. This dataset encompasses information on 81,741 films and was extracted from the Freebase database dated November 4, 2012. Covering the period from 1915 to 2012, it includes a diverse range of historical films. The metadata provides details about movies, such as box office revenue, genre, release period, language, and aligned information about characters and the actors portraying them (name, gender, age, etc.). Additional information about this dataset can be found on the CMU Movie Summary Corpus page: [CMU Movie Summary Corpus]([https://link-url-here.org](https://www.cs.cmu.edu/~ark/personas/)), and in the dataset README, which includes column names: [README.txt]( https://www.cs.cmu.edu/~ark/personas/data/README.txt).

To enhance our dataset and address missing values, we incorporated several additional datasets, including information on ratings and awards:

Initially, we used the MovieLens dataset *INCLUDE DATASET DESCRIPTION*. Furthermore, we integrated data from the IMDb dataset to gather users' ratings. This dataset provides average ratings and the number of votes for more than three million movies, accessible at: [IMDb Dataset](https://developer.imdb.com/non-commercial-datasets/). Finally, for the analysis of awards received by movies, we utilized a dataset derived from the Academy Awards Database — a comprehensive record of Oscar's nominees and winners, encompassing movie names, year, and category, since the first Oscar's to the present day. This dataset can be found at: [Academy Awards Database](https://www.oscars.org/oscars/awards-databases).

## Research Questions:

A list of research questions you would like to address during the project.

- Question 1: Are there any recurring patterns between a film's genre and its release timing within a year? If such patterns exist, do they vary based on the location of the movie, and can these insights contribute to predicting the genre of upcoming film releases in subsequent seasons?
- Question 2: How does release timing impact a movie's success, particularly its likelihood of winning awards?
- Question 3: In what ways is a movie's popularity affected by its release period, considering factors such as box office revenue and ratings?
- Question 4: To what extent do the involvement of specific actors influence a film's success, including its chances of winning awards, expected box office revenue, and ratings? Additionally, can we generate actor groups based on genres and predict their probability of achieving success?

## Methods

- Descriptive Analysis: Explore the historical evolution of genre popularity and seasonal trends.
- Causal Discovery: Identify factors influencing changes in genre popularity.
- Exploratory Analysis: Analyze the correlation between release timing, box office success, and potential awards.

## Proposed Timeline:

- 17.11.2003: Finish milestone P2, start Homework 2
- 24.11.2023: Complete project directions given milestone P2 feedback and initial analyses
- 01.12.2023: Homework 2 deadline, begin final analysis
- 08.12.2023: Draft data story and finish analysis
- 15.12.2023: Finish data story, start website and final review
- 23.12.2023: Milestone P3 deadline

## Organisation within the team:

- Léa: 
- Salim: 
- Jason: 
- Pierre-Hadrien: 
- Yanruiqi: 
