import os
import numpy as np
import pandas as pd
import requests
from tqdm.contrib.concurrent import thread_map
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

links = pd.read_csv("movielens_latest/links.csv")
movies = pd.read_csv("movielens_latest/movies.csv")

def title_fix(nosto_title):
    valo_title = nosto_title.split('(')[0]
    return valo_title[0:len(valo_title)-1]

movies['title'] = movies['title'].apply(title_fix)
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
movies = movies.merge(links, on="movieId")
movies = movies.drop(columns=["imdbId"])

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
api_key = os.getenv("API_KEY")

def get_movie_details(row):
    tmdbId, movieId, genres = row.tmdbId, row.movieId, row.genres
    url = f"https://api.themoviedb.org/3/movie/{tmdbId}?api_key={api_key}&append_to_response=keywords,casts,crews"
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
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
results = thread_map(get_movie_details, movies.itertuples(index=False), max_workers=8)

movies2 = pd.DataFrame(columns=['movieId', 'tag'])
failed_ids = []

for movieId, tag in results:
    if tag is not None:
        movies2 = pd.concat([movies2, pd.DataFrame({'movieId': [int(movieId)], 'tag': [tag]})], ignore_index=True)
    else:
        failed_ids.append(movieId)

movies2.to_csv("upgraded_movielens_latest/movie_w_data.csv", index=False)
print(f"Total failures : {len(failed_ids)}")

with open("failed_ids.txt", "w") as f:
    for failed_id in failed_ids:
        f.write(f"{failed_id}\n")