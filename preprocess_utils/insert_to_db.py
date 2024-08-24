import os
import time
import torch
import joblib
import psycopg2
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from collections import Counter
from scipy.sparse import csr_matrix
from tqdm.autonotebook import tqdm, trange
from psycopg2.extras import execute_values
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


conn = psycopg2.connect(
    dbname = os.getenv("DB_NAME"),
    user = os.getenv("USER"),
    password = os.getenv("PASSWORD"),
    host = os.getenv("HOST"),
    port = os.getenv("PORT")
)
cur = conn.cursor()

print("Started loading data")
start_load = time.time()
Q_b = np.load('bayesian_data/Q_b.npy')

movie_mapper = joblib.load('data/movie_mapper.joblib')
movie_mapper_b = joblib.load('bayesian_data/movie_mapper_b.joblib')
title_2_tmdbID = joblib.load('helper_data/title_2_tmdbID.joblib')

titles = pd.read_csv('upgraded_movielens_latest/titles.csv')
ratings = pd.read_csv('upgraded_movielens_latest/filtered_ratings.csv')
tags = pd.read_csv('upgraded_movielens_latest/upgraded_movies.csv')
end_load = time.time()
print("Completed loading data")
print(f'Total Data Loadung Time : {end_load-start_load}')

unique_movie_ids_rat = ratings['movieId'].unique()

tags_filtered = tags[tags['movieId'].isin(unique_movie_ids_rat)]

tags = tags_filtered

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

model = SentenceTransformer('all-mpnet-base-v2')
model = model.to(device)

def id_to_all(id):
    movie_indices = movie_mapper[id]
    usr_vec = Q_b[movie_indices]
    title = titles.loc[titles['movieId'] == id, 'title'].values[0]
    tmdbId = title_2_tmdbID[title]
    tag = tags.loc[tags['movieId'] == id, 'tag'].values[0]
    with torch.no_grad():
        embeddings = model.encode(tag, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
    return movie_indices, tmdbId, title, tag, usr_vec, embeddings

cur.execute("""
CREATE TABLE IF NOT EXISTS movies_v2 (
    movieId INTEGER PRIMARY KEY,
    movie_indices INTEGER,
    tmdbId integer
    title TEXT,
    tag TEXT,
    usr_vec VECTOR(300),
    embeddings VECTOR(768)
);
""")

insert_query = """
INSERT INTO movies_v2 (movieId, movie_indices, tmdbId, title, tag, usr_vec, embeddings)
VALUES (%s, %s, %s, %s, %s, %s. %s)
ON CONFLICT (movieId) DO UPDATE SET
    movie_indices = EXCLUDED.movie_indices,
    tmdbId = EXCLUDED.tmdbId
    usr_vec = EXCLUDED.usr_vec,
    title = EXCLUDED.title,
    tag = EXCLUDED.tag,
    embeddings = EXCLUDED.embeddings
"""

for i in tqdm(unique_movie_ids_rat):
    movie_indices, tmdbId, title, tag, usr_vec, embeddings = id_to_all(i)
    movie_id = int(i) if isinstance(i, np.integer) else i
    movie_indices = int(movie_indices) if isinstance(movie_indices, np.integer) else movie_indices
    tmdbId = int(tmdbId)
    usr_vec_str = ','.join(map(str, usr_vec))
    embeddings_str = ','.join(map(str, embeddings))
    usr_vec_str = "["+usr_vec_str+"]"
    embeddings_str = "["+embeddings_str+"]"

    try:
        cur.execute(insert_query, (movie_id, movie_indices, tmdbId, title, tag, usr_vec_str, embeddings_str))
        conn.commit()
    except Exception as e:
        print(f"Error inserting movie_id {movie_id}: {str(e)}")
        conn.rollback()
        
cur.close()
conn.close()

print("Data insertion completed.")





