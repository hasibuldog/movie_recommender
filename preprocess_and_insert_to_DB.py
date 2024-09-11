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

load_dotenv(override=True)

db_name = os.getenv("DB_NAME")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")

print("Database connection parameters:")
print(f"Database name: {db_name}")
print(f"User: {user}")
print(f"Host: {host}")
print(f"Port: {port}")
print(f"Password: {'*' * len(password)}")  


conn = psycopg2.connect(
    dbname=db_name,
    user=user,
    password=password,
    host=host,
    port=port
)
print("Successfully connected to the database.")
cur = conn.cursor()

ratings = pd.read_csv('ml-latest/ratings.csv')
links = pd.read_csv("ml-latest/links.csv")
movies_w_title = pd.read_csv("ml-latest/movies.csv")

# def title_fix(nosto_title):
#     valo_title = nosto_title.split('(')[0]
#     return valo_title[0:len(valo_title)-1]

# movies_w_title['title'] = movies_w_title['title'].apply(title_fix)
# movies_w_title['genres'] = movies_w_title['genres'].apply(lambda x: x.split("|"))
# movies_w_title = movies_w_title.merge(links, on="movieId")
# movies_w_title = movies_w_title.drop(columns=["imdbId"])

def process_response(response, movieId):
    movie_data = response.json()
    title = movie_data.get('title')
    tagline = movie_data.get('tagline')
    release_date = movie_data.get('release_date')
    runtime = movie_data.get('runtime')
    genres = [genre['name'] for genre in movie_data.get('genres', [])]
    genres = ' '.join(genres)
    overview = movie_data.get('overview')
    credits = movie_data.get('credits', {})
    director = ', '.join([person['name'] for person in credits.get('crew', []) if person['job'] == 'Director'])
    cast = ', '.join([actor['name'] for actor in credits.get('cast', [])[:5]])  # Top 5 cast members
    poster_url = f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}"
    vote_average = movie_data.get('vote_average')
    popularity = movie_data.get('popularity')

    overview = movie_data.get("overview", "")
    popularity = str(movie_data.get("popularity", ""))
    tagline = movie_data.get("tagline", "")
    keywords = " ".join(keyword["name"] for keyword in movie_data.get("keywords", {}).get("keywords", []))
    casts = " ".join(cast["character"].replace(" ", "") for cast in movie_data.get("casts", {}).get("cast", [])[:5])
    director = next((crew["name"].replace(" ", "") for crew in movie_data.get("casts", {}).get("crew", []) if crew["job"] == "Director"), "")
    vote_average = str(vote_average)
    vote_count = str(movie_data.get("vote_count", ""))
    tag = f"title: {title} genres {genres} overview: {overview} keywords: {keywords} casts: {casts} director: {director} vote_average: {vote_average} vote_count: {vote_count} popularity: {popularity}"
    return movieId, title, tag, genres, overview, cast, tagline, release_date, runtime, poster_url, vote_average, popularity

load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

# # Sample 5000 movies
# movies_sample = links.sample(n=5000, random_state=42)
# print(f"Number of movies to process: {len(movies_sample)}")

status_codes = []

def get_movie_details(row):
    movieId, tmdbId  = row.movieId, row.tmdbId
    url = f"https://api.themoviedb.org/3/movie/{tmdbId}?api_key={api_key}&append_to_response=keywords,casts,crews"
    
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    try:
        response = session.get(url)
        status_codes.append(response.status_code)
        if response.status_code == 200:
            processed_data = process_response(response, movieId)
            return processed_data[0], processed_data[1], processed_data[2], processed_data[3], processed_data[4], processed_data[5], processed_data[6], processed_data[7], processed_data[8], processed_data[9], processed_data[10], processed_data[11]
        else:
            return movieId, None
    except requests.exceptions.RequestException:
        return movieId, None

print("Fetching movie details......")
results = list(thread_map(get_movie_details, links.itertuples(index=False), max_workers=8))
print("Fetching has completed")

# print("Fetching movie details......")
# results = thread_map(get_movie_details, links.itertuples(index=False), max_workers=8)
# print("Fetching has completed")

# Count each type of status code
status_code_counts = {}
for code in status_codes:
    if code in status_code_counts:
        status_code_counts[code] += 1
    else:
        status_code_counts[code] = 1
print("Total status_codes: ", len(status_codes))
print("Status Code Counts:")
for code, count in status_code_counts.items():
    print(f"Status {code}: {count}")


movies_with_all = pd.DataFrame(columns=['movieId', 'tmdbId', 'imdbId', 'title', 'tag', 'genres', 'overview', 'cast', 'tagline', 'release_date', 'runtime', 'poster_url', 'vote_average', 'popularity' ])
failed_ids = []


print(f"Total results: {len(results)}")
print(f"Sample result: {results[0] if results else 'No results'}")
print("Upgrading and Filtering dataframes......")

nulls = []

for result in results:
    if len(result) == 12:
        movieId, title, tag, genres, overview, cast, tagline, release_date, runtime, poster_url, vote_average, popularity = result
        fields = ['movieId', 'title', 'tag', 'genres', 'overview', 'cast', 'tagline', 'release_date', 'runtime', 'poster_url', 'vote_average', 'popularity']
        for field, value in zip(fields, result):
            if value is None:
                nulls.append((field, movieId))        
        if all([movieId, title, tag, genres, overview, poster_url, vote_average, popularity]):
            tmdbId = links.loc[links['movieId'] == movieId, 'tmdbId'].values[0]
            imdbId = links.loc[links['movieId'] == movieId, 'imdbId'].values[0]
            
            movies_with_all = pd.concat([
                movies_with_all, pd.DataFrame({
                    'movieId': [int(movieId)],
                    'tmdbId': [int(tmdbId)],
                    'imdbId': [int(imdbId)],
                    'title': [str(title)],
                    'tag': [str(tag) if tag else ''],
                    'genres': [str(genres)],
                    'overview': [str(overview)],
                    'cast': [str(cast) if cast else ''],
                    'tagline': [str(tagline) if tagline else ''],
                    'release_date': [str(release_date) if release_date else ''],
                    'runtime': [int(runtime) if runtime else 0],
                    'poster_url': [str(poster_url) if poster_url else ''],
                    'vote_average': [float(vote_average) if vote_average else 0.0],
                    'popularity': [float(popularity) if popularity else 0.0]
                })
            ], ignore_index=True)
        else:
            failed_ids.append(movieId)
    else:
        if len(result) > 0:
            failed_ids.append(result[0])

print(f"Failed to fetch details for {len(failed_ids)} movies")
print(f"Shape of movies_with_all: {movies_with_all.shape}")

# Print some sample rows to verify data
print("\nSample rows from movies_with_all:")
print(movies_with_all.head())

# Print some statistics about the data
print("\nData statistics:")
for column in movies_with_all.columns:
    null_count = movies_with_all[column].isnull().sum()
    print(f"{column}: {null_count} null values")

common_ids = set(ratings['movieId']) & set(movies_with_all['movieId'])
print(f"Number of common IDs: {len(common_ids)}")

ratings_filtered = ratings[ratings['movieId'].isin(common_ids)]
movies_with_all_filtered = movies_with_all[movies_with_all['movieId'].isin(common_ids)]
movies_w_title_filtered = movies_w_title[movies_w_title['movieId'].isin(common_ids)]
links_filtered = links[links['movieId'].isin(common_ids)]

print(f"Shape of ratings_filtered: {ratings_filtered.shape}")
print(f"Shape of movies_with_all_filtered: {movies_with_all_filtered.shape}")
print(f"Shape of movies_w_title_filtered: {movies_w_title_filtered.shape}")
print(f"Shape of links_filtered: {links_filtered.shape}")

movies_with_all_filtered.to_csv("upgraded_ml-latest/movies_with_all_filtered.csv", index=False)
ratings_filtered.to_csv("upgraded_ml-latest/ratings_filtered.csv", index=False)
links_filtered.to_csv("upgraded_ml-latest/links_filtered.csv", index=False)
movies_w_title_filtered.to_csv("upgraded_ml-latest/movies_w_title_filtered.csv", index=False)

print("CSV files have been saved for further processing (if needed).")



# movies_with_all_filtered = pd.read_csv("upgraded_ml-latest/movies_with_all_filtered.csv",  engine='python')
# links_filtered = pd.read_csv("upgraded_ml-latest/links_filtered.csv",  engine='python')
# ratings_filtered = pd.read_csv("upgraded_ml-latest/ratings_filtered.csv",  engine='python')

print("\nData statistics: (before null removal)")
for column in movies_with_all_filtered.columns:
    null_count = movies_with_all_filtered[column].isnull().sum()
    print(f"{column}: {null_count} null values")

print(f"Shape of ratings_filtered: {ratings_filtered.shape}")
print(f"Shape of movies_with_all_filtered: {movies_with_all_filtered.shape}")
print(f"Shape of links_filtered: {links_filtered.shape}")

# Drop 'cast' and 'tagline' columns from movies_with_all_filtered
movies_with_all_filtered = movies_with_all_filtered.drop(columns=['cast', 'tagline'])

print(f"Shape of movies_with_all_filtered after dropping 'cast' and 'tagline': {movies_with_all_filtered.shape}")

# Remove rows with any null values
movies_with_all_filtered = movies_with_all_filtered.dropna()

print("Data type of movieId in movies_with_all_filtered:", movies_with_all_filtered['movieId'].dtype)
print("Data type of movieId in ratings_filtered:", ratings_filtered['movieId'].dtype)

movies_with_all_filtered['movieId'] = movies_with_all_filtered['movieId'].astype(int)
ratings_filtered['movieId'] = ratings_filtered['movieId'].astype(int)

movies_ids = set(movies_with_all_filtered['movieId'].unique())
ratings_ids = set(ratings_filtered['movieId'].unique())
common_ids = list(movies_ids & ratings_ids)

print(f"Number of unique movieIds in movies_with_all_filtered: {len(movies_ids)}")
print(f"Number of unique movieIds in ratings_filtered: {len(ratings_ids)}")
print(f"Number of common movieIds: {len(common_ids)}")


print(f"Shape of movies_with_all_filtered after removing null values: {movies_with_all_filtered.shape}")
print(f"Shape of ratings_filtered after removing null values: {ratings_filtered.shape}")
print(f"Shape of links_filtered after removing null values: {links_filtered.shape}")



# Print some statistics about the data
print("\nData statistics: (after null removal)")
for column in movies_with_all_filtered.columns:
    null_count = movies_with_all_filtered[column].isnull().sum()
    print(f"{column}: {null_count} null values")




movies_with_all_filtered = movies_with_all_filtered.sort_values('movieId', ascending=True)

movie_stats = ratings_filtered.groupby('movieId')['rating'].agg(['count', 'mean'])
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings_filtered.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')

ratings_copy = ratings_filtered
ratings = ratings_filtered.drop(columns=['timestamp'])
bayesian_ratings = ratings_copy.merge(movie_stats[["movieId", "bayesian_avg"]], on='movieId')
bayesian_ratings = bayesian_ratings.drop(columns=['timestamp', 'rating'])
bayesian_ratings = bayesian_ratings.sort_values('userId', ascending=True)
bayesian_ratings = bayesian_ratings.rename(columns={'bayesian_avg': 'rating'})

print("Created bayesian average ratings for ratings dataframe")

print(f"Shape of bayesian_ratings: {bayesian_ratings.shape}")
print(f"Number of unique users: {bayesian_ratings['userId'].nunique()}")
print(f"Number of unique movies: {bayesian_ratings['movieId'].nunique()}")

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

X_b, user_mapper_b, movie_mapper_b, user_inv_mapper_b, movie_inv_mapper_b = create_X(bayesian_ratings)

# After creating X_b
print(f"Shape of X_b: {X_b.shape}")

svd = TruncatedSVD(n_components=300, n_iter=10)
Q_b = svd.fit_transform(X_b.T)

print("Created sparse matrix for ratings vs movies and truncated into 300 dimention")

svd = TruncatedSVD(n_components=300, n_iter=10)
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
    movie_indices = int(movie_mapper_b[id])
    usr_vec = Q_b[movie_indices]

    tmdbId = int(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'tmdbId'].values[0])
    imdbId = int(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'imdbId'].values[0])   
    title = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'title'].values[0])
    tag = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'tag'].values[0])
    genres = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'genres'].values[0])
    overview = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'overview'].values[0]   )
    release_date = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'release_date'].values[0]   )
    runtime = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'runtime'].values[0])
    poster_url = str(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'poster_url'].values[0])
    vote_average = float(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'vote_average'].values[0])
    popularity = float(movies_with_all_filtered.loc[movies_with_all_filtered['movieId'] == id, 'popularity'].values[0])

    with torch.no_grad():
        embeddings = model.encode(tag, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()

    return movie_indices, usr_vec, embeddings, tmdbId, imdbId, title, tag, genres, overview, release_date, runtime, poster_url, vote_average, popularity

cur.execute("""
CREATE TABLE IF NOT EXISTS movies_v5 (
    movieId INTEGER PRIMARY KEY,
    movie_indices INTEGER,
    usr_vec VECTOR(300),
    embeddings VECTOR(768),
    tmdbId integer,
    imdbId integer,
    title TEXT,
    tag TEXT,  
    genres TEXT,
    overview TEXT,
    tagline TEXT,
    release_date TEXT,
    runtime TEXT,
    poster_url TEXT,
    vote_average FLOAT,
    popularity FLOAT
);
""")

print("Table Created")

insert_query = """
INSERT INTO movies_v5 (movieId, movie_indices, usr_vec, embeddings, tmdbId, imdbId, title, tag, genres, overview, release_date, runtime, poster_url, vote_average, popularity)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (movieId) DO UPDATE SET
    movie_indices = EXCLUDED.movie_indices,
    usr_vec = EXCLUDED.usr_vec,
    embeddings = EXCLUDED.embeddings,
    tmdbId = EXCLUDED.tmdbId,
    imdbId = EXCLUDED.imdbId,
    title = EXCLUDED.title,
    tag = EXCLUDED.tag,  
    genres =EXCLUDED.genres,
    overview =EXCLUDED.overview,
    tagline =EXCLUDED.tagline,
    release_date =EXCLUDED.release_date,
    runtime =EXCLUDED.runtime,
    poster_url =EXCLUDED.poster_url,
    vote_average =EXCLUDED.vote_average,
    popularity =EXCLUDED.popularity
"""
total_exec = 0
print("Inserting nessecery datas into postgres database")

for movieId in tqdm(common_ids):
    movieId = int(movieId)
    movie_indices, usr_vec, embeddings, tmdbId, imdbId, title, tag, genres, overview, release_date, runtime, poster_url, vote_average, popularity = id_to_all(movieId)
    usr_vec = ','.join(map(str, usr_vec))
    usr_vec = "["+usr_vec+"]"
    embeddings = ','.join(map(str, embeddings))
    embeddings = "["+embeddings+"]"

    try:
        cur.execute(insert_query, (movieId, movie_indices, usr_vec, embeddings, tmdbId, imdbId, title, tag, genres, overview, release_date, runtime, poster_url, vote_average, popularity))
        total_exec += 1
        conn.commit()
    except Exception as e:
        print(f"Error inserting movie_id {movieId}: {str(e)}")
        conn.rollback()

print("Data insertion completed. Total execution : ", total_exec)


cur.execute("""
    CREATE INDEX ON movies_v5 USING ivfflat (usr_vec vector_l2_ops);
""")

print("""
    CREATED INDEX ON movies_v5 USING ivfflat (usr_vec vector_l2_ops);
""")

cur.execute("""
    CREATE INDEX ON movies_v5 USING ivfflat (embeddings vector_cosine_ops);
""")

print("""
    CREATED INDEX ON movies_v5 USING ivfflat (embeddings vector_cosine_ops);
""")

cur.execute("""
    CREATE INDEX idx_movies_v5_title ON movies_v5 USING GIN (to_tsvector('english', title));
""")

print("""
    CREATED INDEX idx_movies_v5_title ON movies_v5 USING GIN (to_tsvector('english', title));
""")

        
cur.close()
conn.close()

print("Data insertion completed.")
print("Database connection closed.")




