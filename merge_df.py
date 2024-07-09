import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ratings = pd.read_csv('movielens_latest/ratings.csv')
movies = pd.read_csv('movielens_latest/movies.csv')
genome_scores = pd.read_csv("movielens_latest/genome-scores.csv")
genome_tags = pd.read_csv("movielens_latest/genome-tags.csv")
genome_data = pd.merge(genome_scores, genome_tags, on='tagId')
movie_id = 1  # for Toy Story (1995)
movie_genome = genome_data[genome_data['movieId'] == movie_id]
top_tags = movie_genome.sort_values(by='relevance', ascending=False).head(30)
bottom_tags = movie_genome.sort_values(by='relevance', ascending=True).head(30)
ratings_with_movies = pd.merge(ratings, movies, on='movieId')
ratings_genome = pd.merge(ratings_with_movies, genome_data, on='movieId')
ratings_genome.head()