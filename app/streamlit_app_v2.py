import os
import requests
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
RECOMMENDATION_URL = "http://localhost:8000/recommend"

title_2_tmdbID = joblib.load('helper_data/title_2_tmdbID.joblib')

@st.cache_data
def fetch_movie_details(tmdb_id):
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,reviews"
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None

@st.cache_data
def get_recommendations(movie_title, num_rec):
    url = RECOMMENDATION_URL
    response = requests.post(url, json={"title_string": movie_title, "n_recommendations": num_rec})
    return response.json()

def display_movie_info(movie, is_main=False):
    col1, col2 = st.columns([1, 2])
    with col1:
        poster_path = movie.get('poster_path')
        if poster_path:
            st.image(f"{POSTER_BASE_URL}{poster_path}", use_column_width=True)
    with col2:
        st.subheader(movie.get('title', 'Unknown Title'))
        st.write(f"**Rating:** {movie.get('vote_average', 'N/A')}/10")
        st.write(f"**Released:** {movie.get('release_date', 'N/A')}")
        genres = movie.get('genres', [])
        if genres:
            st.write("**Genres:** " + ", ".join([genre['name'] for genre in genres]))
        if is_main:
            st.write("**Overview:**")
            st.write(movie.get('overview', 'No overview available.'))

st.title("Movie Recommendation System")

# Initialize session state variables
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'searched_movie' not in st.session_state:
    st.session_state.searched_movie = None

movie_title = st.text_input("Enter a movie title:")
num_rec = st.number_input("Enter preferred number of recommendations:", min_value=10, max_value=50, step=5)

if st.button("Get Recommendations"):
    if movie_title:
        with st.spinner("Fetching recommendations..."):
            try:
                recommendations_data = get_recommendations(movie_title, num_rec)
                if recommendations_data:
                    st.session_state.recommendations = recommendations_data.get('recommendations', [])
                    st.session_state.searched_movie = recommendations_data.get('info')
                    st.session_state.selected_movie = None  # Clear any previously selected movie
                else:
                    st.error("No recommendations received from the API.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if st.session_state.searched_movie:
    st.header("You searched for this movie:")
    searched_movie_id, searched_movie_title = st.session_state.searched_movie
    tmdb_id = title_2_tmdbID.get(searched_movie_title)
    if tmdb_id:
        movie_details = fetch_movie_details(tmdb_id)
        if movie_details:
            display_movie_info(movie_details, is_main=True)
        else:
            st.warning("Failed to fetch movie details from TMDb.")
    else:
        st.warning(f"No TMDb ID found for the movie: {searched_movie_title}")
    
    if st.session_state.recommendations:
        st.header("You might also be interested in these movies:")
        
        cols = st.columns(5)
        for index, movie in enumerate(st.session_state.recommendations):
            with cols[index % 5]:
                tmdb_id = title_2_tmdbID.get(movie['title'])
                if tmdb_id:
                    movie_details = fetch_movie_details(tmdb_id)
                    if movie_details:
                        poster_path = movie_details.get('poster_path')
                        if poster_path:
                            st.image(f"{POSTER_BASE_URL}{poster_path}", use_column_width=True)
                        
                        st.write(f"**{movie_details.get('title', 'Unknown Title')}**")
                        rating = movie_details.get('vote_average')
                        if rating:
                            st.write(f"Rating: {rating}/10")
                        
                        if st.button(f"View Details", key=f"view_{index}"):
                            st.session_state.selected_movie = movie_details
                else:
                    st.write(f"No TMDb ID found for: {movie['title']}")

# Display selected movie details
if st.session_state.selected_movie:
    st.header("Movie Details:")
    display_movie_info(st.session_state.selected_movie, is_main=True)

    if st.button("Back to Recommendations"):
        st.session_state.selected_movie = None

elif not st.session_state.searched_movie:
    st.warning("Please enter a movie title and click 'Get Recommendations'.")