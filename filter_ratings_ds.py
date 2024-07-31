import pandas as pd

ratings = pd.read_csv('movielens_latest/ratings.csv')
movies = pd.read_csv('upgraded_movielens_latest/upgraded_movies.csv', engine='python')

movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')

movies = movies.dropna(subset=['movieId'])

movies['movieId'] = movies['movieId'].astype(int)

valid_movie_ids = set(movies['movieId'])
ratings_filtered = ratings[ratings['movieId'].isin(valid_movie_ids)]

ratings_filtered.to_csv("upgraded_movielens_latest/filtered_ratings.csv", index=False)

common_ids = set(ratings['movieId']) & set(movies['movieId'])
print("\nNumber of common movieIds:", len(common_ids))
