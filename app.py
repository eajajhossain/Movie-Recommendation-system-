import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_data():
    df = pickle.load(open("df.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    indices = pickle.load(open("indices.pkl", "rb"))

    # Normalize indices keys
    indices = {k.lower(): v for k, v in indices.items()}

    # Normalize df titles
    df["title"] = df["title"].str.lower()

    return df, tfidf_matrix, indices


df, tfidf_matrix, indices = load_data()


def recommend_movies(title, n=10):
    title = title.lower()

    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    sim_indices = sim_scores.argsort()[::-1][1:n+1]
    return df.iloc[sim_indices]["title"].tolist()


st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Content-based movie recommender using TF-IDF")

movie_list = sorted(df["title"].str.lower().unique())
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)

    if not recommendations:
        st.error("Movie not found. Your dataset or indices mapping is broken.")
    else:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("👉", movie)
