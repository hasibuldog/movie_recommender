import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from pydantic import BaseModel
from fuzzywuzzy import process
from collections import Counter
from scipy.sparse import csr_matrix
from fastapi import FastAPI, HTTPException
from sklearn.neighbors import NearestNeighbors



print("Started loading data")
start_load = time.time()
Q = np.load('data/Q.npy')
Q_b = np.load('bayesian_data/Q_b.npy')
cosine_sim = np.load('cosine_similarities/cosine_sim.npy')

user_mapper = joblib.load('data/user_mapper.joblib')
movie_mapper = joblib.load('data/movie_mapper.joblib')
user_inv_mapper = joblib.load('data/user_inv_mapper.joblib')
movie_inv_mapper = joblib.load('data/movie_inv_mapper.joblib')

user_mapper_b = joblib.load('bayesian_data/user_mapper_b.joblib')
movie_mapper_b = joblib.load('bayesian_data/movie_mapper_b.joblib')
user_inv_mapper_b = joblib.load('bayesian_data/user_inv_mapper_b.joblib')
movie_inv_mapper_b = joblib.load('bayesian_data/movie_inv_mapper_b.joblib')

title_2_idx = joblib.load('helper_data/title_2_idx.joblib')
title_2_id = joblib.load('helper_data/title_2_id.joblib')
id_2_title = joblib.load('helper_data/id_2_title.joblib')

titles = pd.read_csv('upgraded_movielens_latest/titles.csv')
end_load = time.time()
print("Completed loading data")
print(f'Total Data Loadung Time : {end_load-start_load}')

def movie_finder(title):
    all_titles = titles['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
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

def get_content_based_recommendations(title_string, n_recommendations=15):
    title = movie_finder(title_string)
    print(f'For Movie : {title} \nThese are recommendations...')
    idx = title_2_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    rec = titles['title'].iloc[similar_movies].tolist()
    return rec

def get_collab_based_recommendations_bayesian_shrinked(title_string, n_recommendations=15):
    title = movie_finder(title_string)
    print(f'For Movie : {title} \nThese are recommendations...')
    idx = title_2_idx[title]
    movie_id = movie_inv_mapper[idx]
    similar_movies = find_similar_movies(movie_id, Q_b.T, movie_mapper, movie_inv_mapper, metric='cosine', k=n_recommendations)
    rec = [id_2_title[i] for i in similar_movies]
    return rec

def get_collab_based_recommendations_shrinked(title_string, n_recommendations=15):

    title = movie_finder(title_string)
    print(f'For Movie : {title} \nThese are recommendations...')
    idx = title_2_idx[title]
    movie_id = movie_inv_mapper[idx]
    similar_movies = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, metric='cosine', k=n_recommendations)
    rec = [id_2_title[i] for i in similar_movies]

    return rec

def weighted_avg_merge(w1, w2, rec_content, top_collab_recs, n_recommendations):

    movie_scores = {}
    for i, movie in enumerate(rec_content):
        movie_scores[movie] = movie_scores.get(movie, 0) + w1 * (n_recommendations - i)
    for i, movie in enumerate(top_collab_recs):
        movie_scores[movie] = movie_scores.get(movie, 0) + w2 * (n_recommendations - i)
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True) 
    top_recommendations = [movie for movie, score in sorted_movies[:n_recommendations]]

    return top_recommendations

def voting_merge(rec_bayesian_shrinked, rec_only_shrinked, n_recommendations):
    all_collab = rec_bayesian_shrinked + rec_only_shrinked
    rec_counts = Counter(all_collab)
    sorted_recs = sorted(rec_counts.items(), key=lambda x: (-x[1], x[0]))
    top_collab_recs = [movie for movie, count in sorted_recs[:n_recommendations]]

    return top_collab_recs

def get_content_collab_merged_recommendations(title_string, n_recommendations=15, w1 = 0.6, w2 = 0.4):

    title = movie_finder(title_string)
    idx = title_2_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies_content = [i[0] for i in sim_scores]
    rec_content = titles['title'].iloc[similar_movies_content].tolist()


    movie_id = movie_inv_mapper[idx]
    similar_movies_bayesian_shrinked = find_similar_movies(movie_id, Q_b.T, movie_mapper, movie_inv_mapper, metric='cosine', k=n_recommendations)
    rec_bayesian_shrinked = [id_2_title[i] for i in similar_movies_bayesian_shrinked]

    similar_movies_only_shrinked = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, metric='cosine', k=n_recommendations)
    rec_only_shrinked = [id_2_title[i] for i in similar_movies_only_shrinked]

    top_collab_recs = voting_merge(rec_bayesian_shrinked, rec_only_shrinked, n_recommendations)
    top_recommendations = weighted_avg_merge(w1, w2, rec_content, top_collab_recs, n_recommendations)
    info = (title, movie_id)

    return top_recommendations, info

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
app = FastAPI()
app.json_encoder = NumpyEncoder

# class RecommendationRequest(BaseModel):
#     title: str
#     n_recommendations: int = 15

# class RecommendationResponse(BaseModel):
#     recommendations: list
#     info: tuple

class RecommendationRequest(BaseModel):
    title: str
    n_recommendations: int = 15

class RecommendationResponse(BaseModel):
    recommendations: List[str]
    info: Tuple[str, int]

@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Recommendation API"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        start_rec = time.time()
        top_recommendations, info = get_content_collab_merged_recommendations(
            request.title, request.n_recommendations
        )
        end_rec = time.time()
        print(f'Total Recommendation Computation Time : {end_rec-start_rec}')
        
        top_recommendations = [str(rec) for rec in top_recommendations]
        info = (str(info[0]), int(info[1]))
        
        return RecommendationResponse(recommendations=top_recommendations, info=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)