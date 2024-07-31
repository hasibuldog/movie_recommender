import os
import time
import requests
import numpy as np
import pyarrow as pa
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from fuzzywuzzy import process
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

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

def movie_finder(title):
    all_titles = titles['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    X = X.T
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

X_b, user_mapper_b, movie_mapper_b, user_inv_mapper_b, movie_inv_mapper_b = create_X(bayesian_ratings)

cosine_sim = np.load('cosine_similarities/cosine_sim.npy')

title_2_idx = dict(zip(titles['title'], list(titles.index)))

title_2_id = dict(zip(titles['title'], titles['movieId']))

id_2_title = dict(zip(titles['movieId'], titles['title']))

def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    print(f'For Movie : {title} \nThese are recommendations...')
    idx = title_2_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = []
    for i in sim_scores:
        similar_movies.append(i[0]) 
    rec = titles['title'].iloc[similar_movies].tolist()
    return rec

svd = TruncatedSVD(n_components=100, n_iter=50)
Q = svd.fit_transform(X.T)
Q_b = svd.fit_transform(X_b.T)

def get_collab_based_recommendations_bayesian_shrinked(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    print(f'For Movie : {title} \nThese are recommendations...')
    idx = title_2_idx[title]
    movie_id = movie_inv_mapper[idx]
    similar_movies = find_similar_movies(movie_id, Q_b.T, movie_mapper, movie_inv_mapper, metric='cosine', k=n_recommendations)
    rec = [id_2_title[i] for i in similar_movies]
    return rec

def get_collab_based_recommendations_shrinked(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    print(f'For Movie : {title} \nThese are recommendations...')
    idx = title_2_idx[title]
    movie_id = movie_inv_mapper[idx]
    similar_movies = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, metric='cosine', k=n_recommendations)
    rec = [id_2_title[i] for i in similar_movies]
    return rec

print(get_content_based_recommendations('toy story'))
print(get_collab_based_recommendations_bayesian_shrinked('toy story'))
print(get_collab_based_recommendations_shrinked('toy story'))





