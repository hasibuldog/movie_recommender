import joblib
import pandas as pd
titles = pd.read_csv("upgraded_movielens_latest/titles.csv")
links = pd.read_csv("movielens_latest/links.csv")
links = links.drop(columns=['imdbId'])
valid_movie_ids = set(titles['movieId'])
links = links[links['movieId'].isin(valid_movie_ids)]
links = links.sort_values('movieId', ascending=True)
titles = titles.merge(links, on="movieId")
title_2_tmdbID = dict(zip(titles['title'], titles['tmdbId']))
joblib.dump(title_2_tmdbID, 'helper_data/title_2_tmdbID.joblib')