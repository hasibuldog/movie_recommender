# Movie Recommender

This project is a movie recommendation system that uses the MovieLens dataset, TMDB API, and machine learning techniques to provide personalized movie recommendations. It includes a FastAPI backend for serving recommendations and a Streamlit frontend for user interaction.

## Project Goals

1. Preprocess the MovieLens dataset and enrich it with data from TMDB API
2. Store the processed data in a PostgreSQL database with pgvector support for efficient similarity searches
3. Provide a FastAPI endpoint for movie recommendations
4. Offer a user-friendly Streamlit interface for searching movies and getting recommendations

## Setup Instructions

### Prerequisites

- Docker
- Python 3.8+
- wget

### 1. Set up PostgreSQL with pgvector

Run the following commands to set up a PostgreSQL database with pgvector support using Docker:

```bash
docker pull pgvector/pgvector:pg16
docker volume create pgvector_mov_data
docker run --name pgvector_mov_container \
-e POSTGRES_USER=username_here \
-e POSTGRES_PASSWORD=your_password_here \
-p 5433:5432 \
-v pgvector_mov_data:/var/lib/postgresql/data \
-d pgvector/pgvector:pg16
docker exec -it pgvector_mov_container psql -U bulldogg
CREATE DATABASE mov_db_1;
\c mov_db_1
CREATE EXTENSION vector;
```

### 2. Configure Environment Variables

Create a `.env` file in the project root directory with the following content:

```
DB_NAME=mov_db
USER=username_here
PASSWORD=your_password_here
HOST=localhost
PORT=5433
TMDB_API_KEY=your_tmdb_api_key_here
```

Replace `your_tmdb_api_key_here` with your actual TMDB API key.

### 3. Run the Setup Script

Execute the following command to run the setup script:

```bash
./setup.sh
```

This script will:
1. Download the MovieLens dataset
2. Install required Python packages
3. Preprocess the data and insert it into the PostgreSQL database
4. Start the FastAPI service
5. Launch the Streamlit demo app

## Usage

After running the setup script, you can access:

- The FastAPI endpoint at `http://localhost:8000/recommend` and `http://localhost:8000/search_movies`
- The Streamlit app at `http://localhost:8501`

Use the Streamlit app to search for movies and get personalized recommendations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

