import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuration
TMDB_API_KEY = "76e65305176dfaf750289decfc881e99"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
BASE_URL = "http://localhost:8000"
SEARCH_URL = f"{BASE_URL}/search_movies"
RECOMMEND_URL = f"{BASE_URL}/recommend"

st.set_page_config(page_title="Movie Recommendation App", layout="wide")
st.title("Movie Recommendation App")

@st.cache_data
def fetch_movie_details(tmdb_id):
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits,reviews"}
    response = requests.get(url, params=params)
    print(response.json())
    return response.json() if response.status_code == 200 else None

def display_movie_info(movie, col):
    with col:
        poster_path = movie.get('poster_path')
        if poster_path:
            st.image(f"{POSTER_BASE_URL}{poster_path}", use_column_width=True)
        st.write(f"**{movie.get('title', 'Unknown Title')}**")
        st.write(f"Rating: {movie.get('vote_average', 'N/A')}/10")
        genres = movie.get('genres', [])
        if genres:
            st.write("Genres: " + ", ".join([genre['name'] for genre in genres]))

# Search for movies
search_query = st.text_input("Enter a movie title:")
if search_query:
    response = requests.get(SEARCH_URL, params={"title": search_query, "limit": 10})
    if response.status_code == 200:
        search_results = response.json()
        if search_results:
            selected_movie = st.selectbox(
                "Select a movie:",
                options=search_results,
                format_func=lambda x: x['title']
            )
            if selected_movie:
                st.write(f"Selected: {selected_movie['title']}")

                # Get recommendations
                recommend_response = requests.post(
                    RECOMMEND_URL,
                    json={
                        "movie_id": selected_movie['movieId'],
                        "tmdb_id": selected_movie['tmdbId'],
                        "title": selected_movie['title'],
                        "n_recommendations": 20
                    }
                )

                if recommend_response.status_code == 200:
                    recommendations = recommend_response.json()['recommendations']

                    st.subheader("Recommended Movies")
                    
                    # Create rows of 5 columns each
                    for i in range(0, len(recommendations), 5):
                        cols = st.columns(5)
                        for j, movie in enumerate(recommendations[i:i+5]):
                            movie_details = fetch_movie_details(movie['tmdbId'])
                            if movie_details:
                                display_movie_info(movie_details, cols[j])
                            else:
                                with cols[j]:
                                    st.write(movie['title'])
                                    st.write("Details not available")
                                    st.write(f"TMDB ID: {movie['tmdbId']}")
                        
                        # Add a separator between rows
                        st.write("---")
                else:
                    st.error("Error getting recommendations")
        else:
            st.write("No movies found.")
    else:
        st.error("Error searching for movies")

# Display selected movie details
if 'selected_movie' in locals():
    st.subheader("Selected Movie Details")
    movie_details = fetch_movie_details(selected_movie['tmdbId'])
    if movie_details:
        col1, col2 = st.columns([1, 3])
        display_movie_info(movie_details, col1)
        with col2:
            st.write(f"**Overview:** {movie_details.get('overview', 'No overview available.')}")
    else:
        st.write(f"Movie ID: {selected_movie['movieId']}")
        st.write(f"TMDB ID: {selected_movie['tmdbId']}")
        st.write(f"Title: {selected_movie['title']}")
        st.write(f"Similarity: {selected_movie['similarity']:.4f}")