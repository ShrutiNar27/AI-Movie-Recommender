import os
import pickle
import requests
import streamlit as st
import pandas as pd

# Auto-build model if pkl files don't exist
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    import model  # this will generate the pkl files

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# 🔑 PUT YOUR TMDB API KEY HERE
API_KEY = os.getenv("TMDB_API_KEY")
if API_KEY is None:
    API_KEY = "your_actual_api_key_here"

# Fetch movie details
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    
    poster_path = data.get("poster_path")
    rating = data.get("vote_average")
    
    if poster_path:
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        poster_url = "https://via.placeholder.com/500x750?text=No+Image"
    
    return poster_url, rating

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        poster, rating = fetch_movie_details(movie_id)
        recommended_movies.append((title, poster, rating))

    return recommended_movies

# UI
st.title("🎬 AI Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie:",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    cols = st.columns(5)

    for idx, col in enumerate(cols):
        title, poster, rating = recommendations[idx]
        col.image(poster)
        col.write(f"**{title}**")
        col.write(f"⭐ Rating: {rating}")