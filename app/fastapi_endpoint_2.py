import os
import joblib
import logging
import psycopg2
import pandas as pd
from typing import Any, List
from pydantic import BaseModel
from fuzzywuzzy import process
from psycopg2 import Error as PostgresError
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_name = "mov_db_1"
user = "bulldogg"
password = "21101314"
host = "localhost"
port = 5434

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieMatch(BaseModel):
    movieId: int
    tmdbId: int
    title: str
    similarity: float

class RecommendationRequest(BaseModel):
    movie_id: int
    tmdb_id: int
    title: str
    n_recommendations: int

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    info: tuple

def movie_finder(title_string: str, limit: int = 10) -> List[MovieMatch]:
    query = """
    SELECT movieId, tmdbId, title, similarity(title, %s) AS sim
    FROM movies_v3
    WHERE title %% %s OR title ILIKE %s
    ORDER BY sim DESC
    LIMIT %s;
    """
    try:
        if conn.closed:
            logger.info("Reconnecting to the database")
            conn.connect()
        
        with conn.cursor() as cur:
            cur.execute(query, (title_string, title_string, f"%{title_string}%", limit))
            results = cur.fetchall()
        
        conn.commit()
        return [MovieMatch(movieId=row[0], tmdbId=row[1], title=row[2], similarity=row[3]) for row in results]
    except PostgresError as e:
        logger.error(f"Database error in movie_finder: {str(e)}")
        conn.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in movie_finder: {str(e)}")
        conn.rollback()
        raise

def get_content_based_recommendation(movie_id, k):
    embeddings_query = "SELECT embeddings FROM movies_v3 WHERE movieId = %s;"
    cur.execute(embeddings_query, (movie_id,))
    result = cur.fetchone()

    if result:
        embedding = result[0]
    else:
        print(f"No embedding found for movieId {movie_id}")

    knn_query = """
    SELECT movieId, tmdbId, title, embeddings <-> %s::vector AS distance
    FROM movies_v3
    WHERE movieId != %s  -- Exclude the query movie itself
    ORDER BY embeddings <-> %s::vector
    LIMIT %s;
    """

    cur.execute(knn_query, (embedding, movie_id, embedding, k))
    results = cur.fetchall()
    return results


def get_collaborative_recommendation(movie_id, k):
    usr_vec_query = "SELECT usr_vec FROM movies_v3 WHERE movieId = %s;"
    cur.execute(usr_vec_query, (movie_id,))
    result = cur.fetchone()
    if result:
        usr_vec = result[0]
    else:
        print(f"No usr_vec found for movieId {movie_id}")

    query = """
    SELECT movieId, tmdbId, title, usr_vec <-> %s::vector AS distance
    FROM movies_v3
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
    content_distances = [rec[3] for rec in content_recs]
    collab_distances = [rec[3] for rec in collab_recs]
    
    norm_content_distances = min_max_normalize(content_distances)
    norm_collab_distances = min_max_normalize(collab_distances)
    
    content_recs_norm = [(rec[0], rec[1], rec[2], norm_dist) for rec, norm_dist in zip(content_recs, norm_content_distances)]
    collab_recs_norm = [(rec[0], rec[1], rec[2], norm_dist) for rec, norm_dist in zip(collab_recs, norm_collab_distances)]
    rec_dict = {}
    for rec in content_recs_norm + collab_recs_norm:
        movie_id, tmdbId, title, distance = rec
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
                'tmdbId': tmdbId,
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
            'tmdbId': data['tmdbId'],
            'title': data['title'],
            'avg_distance': avg_distance,
            'count': data['count'],
            'content_distance': data['content_distance'],
            'collaborative_distance': data['collaborative_distance']
        })
    print("merged_recs_unsorted", merged_recs)
    merged_recs.sort(key=lambda x: (-x['count'], x['avg_distance']))
    print("merged_recs_sorted", merged_recs)

    return merged_recs[:k]


def get_hybrid_recommendations(movie_id: int, tmdb_id: int, title: str, k: int):
    info = (movie_id, tmdb_id, title)
    content_recs = get_content_based_recommendation(movie_id=movie_id, k=k)
    collab_recs = get_collaborative_recommendation(movie_id=movie_id, k=k)
    merged_recs = merge_recommendations(content_recs, collab_recs, k)
    return merged_recs, info


@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Recommendation API"}

@app.get("/search_movies", response_model=List[MovieMatch])
async def search_movies(title: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=100)):
    try:
        logger.info(f"Searching for movies with title: {title}, limit: {limit}")
        results = movie_finder(title, limit)
        logger.info(f"Found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching movies: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        top_recommendations, info = get_hybrid_recommendations(
            request.movie_id, request.tmdb_id, request.title, request.n_recommendations
        )
        return RecommendationResponse(recommendations=top_recommendations, info=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
