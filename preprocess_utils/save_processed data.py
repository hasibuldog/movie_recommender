import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib

ratings = pd.read_csv('upgraded_movielens_latest/filtered_ratings.csv', engine='pyarrow')
movies = pd.read_csv('upgraded_movielens_latest/upgraded_movies.csv', engine='pyarrow')
movies_w_title = pd.read_csv('movielens_latest/movies.csv', engine='pyarrow')

movies_w_title = movies_w_title.drop(columns=['genres'])
valid_movie_ids = set(movies['movieId'])
titles = movies_w_title[movies_w_title['movieId'].isin(valid_movie_ids)]
movies['movieId'] = movies['movieId'].astype(int)

movies = movies.sort_values('movieId', ascending=True)
titles = titles.sort_values('movieId', ascending=True)

movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')

ratings_copy = ratings
ratings = ratings.drop(columns=['timestamp'])
bayesian_ratings = ratings_copy.merge(movie_stats[["movieId", "bayesian_avg"]], on='movieId')
bayesian_ratings = bayesian_ratings.drop(columns=['timestamp', 'rating'])
bayesian_ratings = bayesian_ratings.sort_values('userId', ascending=True)
bayesian_ratings = bayesian_ratings.rename(columns={'bayesian_avg': 'rating'})

def create_X(df):
    
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))
    
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
X_b, user_mapper_b, movie_mapper_b, user_inv_mapper_b, movie_inv_mapper_b = create_X(bayesian_ratings)

cosine_sim = np.load('cosine_similarities/cosine_sim.npy')

title_2_idx = dict(zip(titles['title'], list(titles.index)))
title_2_id = dict(zip(titles['title'], titles['movieId']))
id_2_title = dict(zip(titles['movieId'], titles['title']))

svd = TruncatedSVD(n_components=100, n_iter=50)
Q = svd.fit_transform(X.T)
Q_b = svd.fit_transform(X_b.T)



np.save('data/Q.npy', Q)
joblib.dump(user_mapper, 'data/user_mapper.joblib')
joblib.dump(movie_mapper, 'data/movie_mapper.joblib')
joblib.dump(user_inv_mapper, 'data/user_inv_mapper.joblib')
joblib.dump(movie_inv_mapper, 'data/movie_inv_mapper.joblib')
titles.to_csv('upgraded_movielens_latest/titles.csv', index=False)

np.save('bayesian_data/Q_b.npy', Q_b)
joblib.dump(user_mapper, 'bayesian_data/user_mapper_b.joblib')
joblib.dump(movie_mapper, 'bayesian_data/movie_mapper_b.joblib')
joblib.dump(user_inv_mapper, 'bayesian_data/user_inv_mapper_b.joblib')
joblib.dump(movie_inv_mapper, 'bayesian_data/movie_inv_mapper_b.joblib')

joblib.dump(title_2_idx, 'helper_data/title_2_idx.joblib')
joblib.dump(title_2_id, 'helper_data/title_2_id.joblib')
joblib.dump(id_2_title, 'helper_data/id_2_title.joblib')