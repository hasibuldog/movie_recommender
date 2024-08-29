import os
import time
import torch
import joblib
import psycopg2
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from tqdm.contrib.concurrent import thread_map
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

load_dotenv()

db_name = os.getenv("DB_NAME")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")

try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )
    print("Successfully connected to the database.")
    cur = conn.cursor()
except Exception as e:
    print(f"An error occurred: {e}")

ratings = pd.read_csv('ml-latest/ratings.csv')
links = pd.read_csv("ml-latest/links.csv")
movies_w_title = pd.read_csv("ml-latest/movies.csv")

def title_fix(nosto_title):
    valo_title = nosto_title.split('(')[0]
    return valo_title[0:len(valo_title)-1]

movies_w_title['title'] = movies_w_title['title'].apply(title_fix)
movies_w_title['genres'] = movies_w_title['genres'].apply(lambda x: x.split("|"))
movies_w_title = movies_w_title.merge(links, on="movieId")
movies_w_title = movies_w_title.drop(columns=["imdbId"])

def process_response(response):
    data = response.json()
    overview = data.get("overview", "")
    popularity = str(data.get("popularity", ""))
    tagline = data.get("tagline", "")
    keywords = " ".join(keyword["name"] for keyword in data.get("keywords", {}).get("keywords", []))
    casts = " ".join(cast["character"].replace(" ", "") for cast in data.get("casts", {}).get("cast", [])[:5])
    director = next((crew["name"].replace(" ", "") for crew in data.get("casts", {}).get("crew", []) if crew["job"] == "Director"), "")
    vote_average = str(data.get("vote_average", ""))
    vote_count = str(data.get("vote_count", ""))
    tag = f"overview: {overview} tagline: {tagline} keywords: {keywords} casts: {casts} director: {director} vote_average: {vote_average} vote_count: {vote_count} popularity: {popularity}"
    return tag

load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

def get_movie_details(row):
    tmdbId, movieId, genres = row.tmdbId, row.movieId, row.genres
    url = f"https://api.themoviedb.org/3/movie/{tmdbId}?api_key={api_key}&append_to_response=keywords,casts,crews"
    
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    try:
        response = session.get(url)
        if response.status_code == 200:
            genres_str = ' Genres : ' + ' '.join(genres)
            return movieId, process_response(response) + genres_str
        else:
            return movieId, None
    except requests.exceptions.RequestException:
        return movieId, None

print("Fetching movie details...")
results = thread_map(get_movie_details, movies_w_title.itertuples(index=False), max_workers=8)

movies_with_tags = pd.DataFrame(columns=['movieId', 'tag'])
failed_ids = []

for movieId, tag in results:
    if tag is not None:
        movies_with_tags = pd.concat([movies_with_tags, pd.DataFrame({'movieId': [int(movieId)], 'tag': [tag]})], ignore_index=True)
    else:
        failed_ids.append(movieId)



movies_with_tags['movieId'] = pd.to_numeric(movies_with_tags['movieId'], errors='coerce')

movies_with_tags = movies_with_tags.dropna(subset=['movieId'])

movies_with_tags['movieId'] = movies_with_tags['movieId'].astype(int)


common_ids = set(ratings['movieId']) & set(movies_with_tags['movieId'])

ratings_filtered = ratings[ratings['movieId'].isin(common_ids)]
movies_with_tags_filtered = movies_with_tags[movies_with_tags['movieId'].isin(common_ids)]
movies_w_title_filtered = movies_w_title[movies_w_title['movieId'].isin(common_ids)]
links_filtered = links[links['movieId'].isin(common_ids)]

movies_with_tags_filtered.to_csv("upgraded_ml-latest/movies_with_tags_filtered.csv", index=False)
ratings_filtered.to_csv("upgraded_ml-latest/ratings_filtered.csv", index=False)
links_filtered.to_csv("upgraded_ml-latest/links_filtered.csv", index=False)
movies_w_title_filtered.to_csv("upgraded_ml-latest/movies_w_title_filtered.csv", index=False)

ratings = ratings_filtered
movies = movies_with_tags_filtered
titles = movies_w_title_filtered
titles = titles.drop(columns=['genres'])

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

title_2_idx = dict(zip(titles['title'], list(titles.index)))
title_2_id = dict(zip(titles['title'], titles['movieId']))
id_2_title = dict(zip(titles['movieId'], titles['title']))
id_2_tmdbID = dict(zip(links_filtered['movieId'], links_filtered['tmdbId']))

svd = TruncatedSVD(n_components=300, n_iter=10)
Q = svd.fit_transform(X.T)
Q_b = svd.fit_transform(X_b.T)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

model = SentenceTransformer('all-mpnet-base-v2')
model = model.to(device)

def id_to_all(id):
    movie_indices = movie_mapper_b[id]
    usr_vec = Q_b[movie_indices]
    title = titles.loc[titles['movieId'] == id, 'title'].values[0]
    tmdbId = links.loc[links['movieId'] == id, 'tmdbId'].values[0]
    imdbId = links.loc[links['movieId'] == id, 'imdbId'].values[0]
    tmdbId = id_2_tmdbID[id]
    tag = movies.loc[movies['movieId'] == id, 'tag'].values[0]
    with torch.no_grad():
        embeddings = model.encode(tag, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
    return movie_indices, tmdbId, imdbId, title, tag, usr_vec, embeddings

cur.execute("""
CREATE TABLE IF NOT EXISTS movies_v3 (
    movieId INTEGER PRIMARY KEY,
    movie_indices INTEGER,
    tmdbId integer,
    imdbId integer,
    title TEXT,
    tag TEXT,
    usr_vec VECTOR(300),
    embeddings VECTOR(768)
);
""")

insert_query = """
INSERT INTO movies_v3 (movieId, movie_indices, tmdbId, imdbId, title, tag, usr_vec, embeddings)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (movieId) DO UPDATE SET
    movie_indices = EXCLUDED.movie_indices,
    tmdbId = EXCLUDED.tmdbId,
    imdbId = EXCLUDED.imdbId,
    usr_vec = EXCLUDED.usr_vec,
    title = EXCLUDED.title,
    tag = EXCLUDED.tag,
    embeddings = EXCLUDED.embeddings
"""

for i in tqdm(common_ids):
    movie_indices, tmdbId, imdbId, title, tag, usr_vec, embeddings = id_to_all(i)
    movie_id = int(i) if isinstance(i, np.integer) else i
    movie_indices = int(movie_indices) if isinstance(movie_indices, np.integer) else movie_indices
    tmdbId = int(tmdbId) if isinstance(tmdbId, np.integer) else tmdbId
    imdbId = int(imdbId) if isinstance(imdbId, np.integer) else imdbId
    usr_vec_str = ','.join(map(str, usr_vec))
    embeddings_str = ','.join(map(str, embeddings))
    usr_vec_str = "["+usr_vec_str+"]"
    embeddings_str = "["+embeddings_str+"]"

    try:
        cur.execute(insert_query, (movie_id, movie_indices, tmdbId, imdbId, title, tag, usr_vec_str, embeddings_str))
        conn.commit()
    except Exception as e:
        print(f"Error inserting movie_id {movie_id}: {str(e)}")
        conn.rollback()


cur.execute("""
    CREATE INDEX ON movies_v3 USING ivfflat (usr_vec vector_l2_ops);
""")
cur.execute("""
    CREATE INDEX ON movies_v3 USING ivfflat (embeddings vector_cosine_ops);
""")
cur.execute("""
    CREATE INDEX idx_movies_v3_title ON movies_v3 USING GIN (to_tsvector('english', title));
""")
        
cur.close()
conn.close()

print("Data insertion completed.")




