import requests
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Constants ---
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_FILENAME = 'movie_embeddings.npy'
METADATA_FILENAME = 'movie_metadata.parquet'
MIN_QUERY_WORDS = 3
MIN_SIMILARITY_THRESHOLD = 0.3
TOP_N_RECOMMENDATIONS = 10

# --- Caching Functions for Loading ---

# Cache the model loading
@st.cache_resource
def load_model(model_name=MODEL_NAME):
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        return None

# --- Function to fetch poster URL from TMDB API ---
@st.cache_data(ttl="1d")
def get_movie_details_from_api(movie_id, api_key):
    """Fetches movie details (including poster path) from TMDB API."""
    if not api_key:
        print("API Key missing, cannot fetch details.")
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"API request failed for movie ID {movie_id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing API response for movie ID {movie_id}: {e}")
        return None
    
# Cache the data loading (embeddings and metadata)
@st.cache_data
def load_data(embeddings_file=EMBEDDINGS_FILENAME, metadata_file=METADATA_FILENAME):
    """Loads the movie embeddings and metadata."""
    if not os.path.exists(embeddings_file):
        st.error(f"Error: Embeddings file not found at {embeddings_file}")
        return None, None
    if not os.path.exists(metadata_file):
        st.error(f"Error: Metadata file not found at {metadata_file}")
        return None, None

    try:
        embeddings = np.load(embeddings_file)
        metadata = pd.read_parquet(metadata_file)
        print("Embeddings and metadata loaded successfully.")

        # Basic validation
        if embeddings.shape[0] != len(metadata):
             st.error("Mismatch between number of embeddings and metadata rows!")
             return None, None

        return embeddings, metadata
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None

# --- Recommendation Function ---

def recommend_movies_by_query(user_query,
                              model,
                              embeddings,
                              metadata,
                              top_n=TOP_N_RECOMMENDATIONS,
                              min_length=MIN_QUERY_WORDS,
                              min_similarity=MIN_SIMILARITY_THRESHOLD):
    """
    Recommends movies based on cosine similarity between user query text
    embedding and pre-computed movie embeddings. Includes basic validation.

    Args:
        user_query (str): The user's text description of desired movie type.
        model (SentenceTransformer): The loaded sentence transformer model.
        embeddings (np.ndarray): The pre-computed movie embeddings array.
        metadata (pd.DataFrame): The metadata DataFrame corresponding to embeddings.
        top_n (int): Number of recommendations to return.
        min_length (int): Minimum number of words required in the query.
        min_similarity (float): Minimum similarity score for the top match.

    Returns:
        tuple: (pd.DataFrame or None, str or None) -
               Returns a DataFrame of recommended movies if successful,
               otherwise None. Also returns a status/error message string or None.
    """
    status_message = None

    # 1. Input Validation
    if not user_query or not isinstance(user_query, str):
        status_message = "Validation Error: Please enter a description."
        return None, status_message
    query_word_count = len(user_query.split())
    if query_word_count < min_length:
        status_message = f"Validation Error: Query too short ({query_word_count} words). Please use at least {min_length} words."
        return None, status_message

    if model is None or embeddings is None or metadata is None:
         status_message = "Error: Core components (model/data) not loaded."
         return None, status_message

    # 2. Encode User Query
    try:
        query_embedding = model.encode([user_query])
    except Exception as e:
        status_message = f"Error encoding your query: {e}"
        return None, status_message

    # 3. Calculate Similarities
    try:
        cosine_scores = cosine_similarity(query_embedding, embeddings)
        similarity_scores = cosine_scores[0]
    except Exception as e:
        status_message = f"Error calculating similarities: {e}"
        return None, status_message

    # 4. Relevance Check
    max_similarity = np.max(similarity_scores)
    if max_similarity < min_similarity:
        status_message = (f"No sufficiently relevant movies found. "
                          f"Highest similarity: {max_similarity:.2f} "
                          f"(Threshold: {min_similarity:.2f}). Try a different description.")
        return None, status_message

    # 5. Get Top N Recommendations
    try:
        top_indices = np.argpartition(similarity_scores, -top_n)[-top_n:]
        sorted_top_indices = top_indices[np.argsort(similarity_scores[top_indices])][::-1]

        recommended_movies = metadata.iloc[sorted_top_indices].copy()
        recommended_movies['similarity'] = similarity_scores[sorted_top_indices]

        status_message = f"Found {len(recommended_movies)} recommendations."
        return recommended_movies, status_message

    except Exception as e:
        status_message = f"Error retrieving top recommendations: {e}"
        return None, status_message
    

