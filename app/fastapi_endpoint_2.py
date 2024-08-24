import os
import joblib
import psycopg2
import pandas as pd
from pydantic import BaseModel
from fuzzywuzzy import process
from typing import List, Dict, Tuple, Any
from fastapi import FastAPI, HTTPException

title_2_id = joblib.load('helper_data/title_2_id.joblib')
title_2_tmdbID = joblib.load('helper_data/title_2_tmdbID.joblib')
titles = pd.read_csv('upgraded_movielens_latest/titles.csv')

print(os.getenv("DB_NAME"))
print(os.getenv("USER"))
print(os.getenv("PASSWORD"))
print(os.getenv("HOST"))
print(os.getenv("PORT"))

conn = psycopg2.connect(
    dbname = os.getenv("DB_NAME"),
    user = os.getenv("USER"),
    password = os.getenv("PASSWORD"),
    host = os.getenv("HOST"),
    port = os.getenv("PORT")
)
cur = conn.cursor()

all_titles = titles['title'].tolist()

def movie_finder(title):
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

def get_content_based_recommendation(movie_id, k):
    embeddings_query = "SELECT embeddings FROM movies WHERE movieId = %s;"
    cur.execute(embeddings_query, (movie_id,))
    result = cur.fetchone()

    if result:
        embedding = result[0]
    else:
        print(f"No embedding found for movieId {movie_id}")

    knn_query = """
    SELECT movieId, title, embeddings <-> %s::vector AS distance
    FROM movies
    WHERE movieId != %s  -- Exclude the query movie itself
    ORDER BY embeddings <-> %s::vector
    LIMIT %s;
    """

    cur.execute(knn_query, (embedding, movie_id, embedding, k))
    results = cur.fetchall()
    return results


def get_collaborative_recommendation(movie_id, k):
    usr_vec_query = "SELECT usr_vec FROM movies WHERE movieId = %s;"
    cur.execute(usr_vec_query, (movie_id,))
    result = cur.fetchone()
    if result:
        usr_vec = result[0]
    else:
        print(f"No usr_vec found for movieId {movie_id}")

    query = """
    SELECT movieId, title, usr_vec <-> %s::vector AS distance
    FROM movies
    WHERE movieId != %s  -- Exclude the query movie itself
    ORDER BY usr_vec <-> %s::vector
    LIMIT %s;
    """

    cur.execute(query, (usr_vec, movie_id, usr_vec, k))
    results = cur.fetchall()
    return results

def min_max_normalize(distances):
    min_dist = min(distances)
    max_dist = max(distances)
    if min_dist == max_dist:
        return [1.0 for _ in distances]
    return [(d - min_dist) / (max_dist - min_dist) for d in distances]

def merge_recommendations(content_recs, collab_recs, k):
    content_distances = [rec[2] for rec in content_recs]
    collab_distances = [rec[2] for rec in collab_recs]
    
    norm_content_distances = min_max_normalize(content_distances)
    norm_collab_distances = min_max_normalize(collab_distances)
    
    content_recs_norm = [(rec[0], rec[1], norm_dist) for rec, norm_dist in zip(content_recs, norm_content_distances)]
    collab_recs_norm = [(rec[0], rec[1], norm_dist) for rec, norm_dist in zip(collab_recs, norm_collab_distances)]
    rec_dict = {}
    for rec in content_recs_norm + collab_recs_norm:
        movie_id, title, distance = rec
        if movie_id in rec_dict:
            rec_dict[movie_id]['distance_sum'] += distance
            rec_dict[movie_id]['count'] += 1
            if rec in content_recs_norm:
                rec_dict[movie_id]['content_distance'] = distance
            if rec in collab_recs_norm:
                rec_dict[movie_id]['collaborative_distance'] = distance
        else:
            rec_dict[movie_id] = {
                'title': title, 
                'distance_sum': distance, 
                'count': 1,
                'content_distance': distance if rec in content_recs_norm else None,
                'collaborative_distance': distance if rec in collab_recs_norm else None
            }
    merged_recs = []
    for movie_id, data in rec_dict.items():
        avg_distance = data['distance_sum'] / data['count']
        merged_recs.append({
            'movie_id': movie_id,
            'title': data['title'],
            'avg_distance': avg_distance,
            'count': data['count'],
            'content_distance': data['content_distance'],
            'collaborative_distance': data['collaborative_distance']
        })
    merged_recs.sort(key=lambda x: (-x['count'], x['avg_distance']))

    return merged_recs[:k]

def get_hybrid_recommendations(title_string, k):
    title = movie_finder(title_string)
    id = title_2_id[title]
    info = (id, title)
    content_recs = get_content_based_recommendation(movie_id=id, k=k)
    collab_recs = get_collaborative_recommendation(movie_id=id, k=k)
    merged_recs = merge_recommendations(content_recs, collab_recs, k)
    return merged_recs, info

print(get_hybrid_recommendations("spiderman", 5))
    
app = FastAPI()

class RecommendationRequest(BaseModel):
    title_string: str
    n_recommendations: int

class RecommendationResponse(BaseModel):
    recommendations: Any
    info: Any

@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Recommendation API"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    print('sex')
    try:
        top_recommendations, info = get_hybrid_recommendations(
            request.title_string, request.n_recommendations
        )
        print(top_recommendations)
        print("##########")
        print(info)
        
        return RecommendationResponse(recommendations=top_recommendations, info=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)