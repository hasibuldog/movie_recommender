import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def fetch_movie_details(tmdb_id):
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,reviews"
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None

# Function to get recommendations (replace this with your actual API call)
@st.cache_data
def get_recommendations(movie_title, num_rec, w1, w2):
    url = "http://localhost:8000/recommend"
    response = requests.post(url, json={"title": movie_title, "n_recommendations": num_rec, "w1": w1, "w2": w2})
    return response.json()

# Function to display movie info
# def display_movie_info(movie, is_main=False):
#     if isinstance(movie, list) and len(movie) > 0:
#         movie = movie[0]  # Take the first item if it's a list
    
#     if not isinstance(movie, dict):
#         st.warning(f"Unexpected movie data type: {type(movie)}")
#         return

#     col1, col2 = st.columns([1, 2])
#     with col1:
#         poster_path = movie.get('poster_path')
#         if poster_path:
#             st.image(f"{POSTER_BASE_URL}{poster_path}", use_column_width=True)
#     with col2:
#         st.subheader(movie.get('title', 'Unknown Title'))
#         st.write(f"**Rating:** {movie.get('vote_average', 'N/A')}/10")
#         st.write(f"**Released:** {movie.get('release_date', 'N/A')}")
#         genres = movie.get('genres', [])
#         if genres:
#             st.write("**Genres:** " + ", ".join([genre.get('name', '') for genre in genres if isinstance(genre, dict)]))
#         if is_main:
#             st.write("**Overview:**")
#             st.write(movie.get('overview', 'No overview available.'))

def display_movie_info(movie, is_main=False):
    # Check if the input is a list and take the first item if it is
    if isinstance(movie, list) and len(movie) > 0:
        movie = movie[0]
    
    # Check if the input is a dictionary
    if not isinstance(movie, dict):
        st.warning(f"Unexpected movie data type: {type(movie)}")
        return

    # Display movie poster, title, rating, release date, and genres
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
            st.write("**Genres:** " + ", ".join([genre.get('name', '') for genre in genres if isinstance(genre, dict)]))
        if is_main:
            st.write("**Overview:**")
            st.write(movie.get('overview', 'No overview available.'))

st.title("Movie Recommendation System")

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'searched_movie' not in st.session_state:
    st.session_state.searched_movie = None

movie_title = st.text_input("Enter a movie title:")

num_rec = st.number_input("Enter prefered number of recommendation: ", min_value=10, max_value=50, step=5)

w1 = st.slider('Select a value', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
w2 = 1 - w1
# Display the selected value
st.write(f'Content weight: {w1}')
st.write(f'Collab weight: {w2}')

if st.button("Get Recommendations"):
    if movie_title:
        with st.spinner("Fetching recommendations..."):
            try:
                recommendations_data = get_recommendations(movie_title, num_rec, w1, w2)
                # st.write("API Response:", json.dumps(recommendations_data, indent=2))  # Debug print
                
                if recommendations_data:
                    st.session_state.recommendations = recommendations_data.get('recommendations', [])
                    st.session_state.searched_movie = recommendations_data.get('info')
                    st.session_state.selected_movie = None  # Clear any previously selected movie
                else:
                    st.error("No recommendations received from the API.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display searched movie and recommendations
# if st.session_state.searched_movie:
    


if st.session_state.searched_movie:
    st.header("You searched for this movie:")
    movie_details = fetch_movie_details(st.session_state.searched_movie.get("tmdbID"))

    print("###\n###\n###")
    print("Fetched Title:  ",movie_details.get('title'))
    print("###\n###\n###")
    print(st.session_state.searched_movie.get("title"), st.session_state.searched_movie.get("tmdbID"))
    print("###\n###\n###")

    if movie_details:
        display_movie_info(movie_details, is_main=True)
    else:
        st.warning("Failed to fetch movie details from TMDb.")
    
    if st.session_state.recommendations:
        st.header("You might also be interested in these movies:")
        
        # Display recommendations in a grid
        cols = st.columns(6)
        for index, movie in enumerate(st.session_state.recommendations):
            with cols[index % 6]:
                movie_details = fetch_movie_details(movie.get('tmdbID'))
                if movie_details:
                    # Display poster
                    poster_path = movie_details.get('poster_path')
                    if poster_path:
                        st.image(f"{POSTER_BASE_URL}{poster_path}", use_column_width=True)
                    
                    # Display title and rating
                    st.write(f"**{movie_details.get('title', 'Unknown Title')}**")
                    rating = movie_details.get('vote_average')
                    if rating:
                        st.write(f"Rating: {rating}/10")
                    
                    # Button to view details
                    if st.button(f"View Details", key=f"view_{index}"):
                        st.session_state.selected_movie = movie_details

# Display selected movie details
if st.session_state.selected_movie:
    
    st.header("Movie Details:")

    display_movie_info(st.session_state.selected_movie, is_main=True)

    if st.button("Back to Recommendations"):
        st.session_state.selected_movie = None

elif not st.session_state.searched_movie:
    st.warning("Please enter a movie title and click 'Get Recommendations'.")