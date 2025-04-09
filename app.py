import streamlit as st
import pandas as pd
from recommender_logic import load_model, load_data, recommend_movies_by_query, get_movie_details_from_api

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood Based Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Inject Custom CSS ---
POSTER_HEIGHT_PX = 400
BORDER_RADIUS_PX = 22

css = f"""
<style>
    .movie-poster-container {{
        width: 100%;
        height: {POSTER_HEIGHT_PX}px; /* Set the fixed height */
        border-radius: {BORDER_RADIUS_PX}px; /* Apply rounded corners to container */
        overflow: hidden; /* IMPORTANT: Makes border-radius clip the image */
        display: flex; /* Use flexbox to help center the image */
        justify-content: center; /* Center image horizontally */
        align-items: center; /* Center image vertically */
        # background-color: #222; /* Optional: dark bg for empty space with 'contain' */
        margin-bottom: 5px;
    }}
    .movie-poster-container img {{
        display: block; /* Prevents potential extra space below image */
        max-width: 100%; /* Ensure image doesn't exceed container width */
        max-height: 100%; /* Ensure image doesn't exceed container height */
        width: auto; /* Maintain aspect ratio */
        height: auto; /* Maintain aspect ratio */
        object-fit: cover; /* Fit entire image within container, no cropping */
        border-radius: {BORDER_RADIUS_PX}px; /* Optional: Apply to image too for smoother edges if container bg shows */
    }}
    .movie-caption {{
        text-align: center;
        font-size: 0.9em;
        margin-bottom: 10px;
    }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
# --- End CSS Injection ---


TMDB_API_KEY = st.secrets.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not found in secrets.toml. Cannot fetch posters.")

# --- Load Model and Data ---
model = load_model()
embeddings, metadata = load_data()

# --- App Title and Description ---
st.title("🎬 Mood-Based Movie Recommender 🍿")
st.write("""
Describe your current mood or the kind of movie plot you're looking for,
and we'll suggest some films! Enter a few words (e.g., "uplifting story about friendship",
"dark and gritty crime thriller", "funny feel-good animation") and click "Recommend".
""")
user_query = st.text_input("Describe your desired movie mood or plot:", "")

# --- Recommendation Button and Logic ---
if st.button("Recommend Movies"):
    if model is not None and embeddings is not None and metadata is not None:
        if user_query:
            st.write(f"Searching for movies similar to: \"{user_query}\"...")
            recommendations_df, status_message = recommend_movies_by_query(
                user_query=user_query,
                model=model,
                embeddings=embeddings,
                metadata=metadata
            )

            if status_message:
                if "Error" in status_message or "No sufficiently relevant" in status_message:
                     st.warning(status_message)
                else:
                     st.info(status_message)

            if recommendations_df is not None and not recommendations_df.empty:
                st.subheader("Top Recommendations:")

                poster_base_url = "https://image.tmdb.org/t/p/w342"
                num_cols = 4
                cols = st.columns(num_cols)

                for display_index, row in enumerate(recommendations_df.itertuples(index=False)):
                    col_index = display_index % num_cols
                    with cols[col_index]:
                        try:
                            movie_id = row.id
                        except AttributeError:
                            st.error("Could not access movie ID.")
                            continue

                        # --- Fetch details via API ---
                        movie_details = get_movie_details_from_api(movie_id, TMDB_API_KEY)
                        poster_path = None
                        if movie_details:
                            poster_path = movie_details.get('poster_path')

                        # --- Get other details ---
                        title = getattr(row, 'title', 'N/A')
                        release_year = getattr(row, 'release_year', 'N/A')
                        rating = getattr(row, 'weighted_rating', 'N/A')
                        similarity = getattr(row, 'similarity', 'N/A')

                        # --- Construct HTML for Poster and Caption ---
                        if poster_path:
                            full_poster_url = poster_base_url + poster_path
                            st.markdown(
                                f'<div class="movie-poster-container"><img src="{full_poster_url}" alt="{title} Poster"></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="movie-poster-container" style="border:1px dashed gray; display:flex; align-items:center; justify-content:center;">'
                                f'<p style="text-align:center; font-size:0.8em; color:gray;">No Poster</p></div>',
                                unsafe_allow_html=True
                            )
                        # Display caption below the container
                        st.markdown(f'<p class="movie-caption">{title} ({release_year})</p>', unsafe_allow_html=True)

                        # --- Display Rating and Similarity ---
                        try:
                            rating_str = f"{float(rating):.2f}" if rating != 'N/A' else 'N/A'
                        except (ValueError, TypeError): rating_str = 'N/A'
                        try:
                            sim_str = f"{float(similarity):.3f}" if similarity != 'N/A' else 'N/A'
                        except (ValueError, TypeError): sim_str = 'N/A'

                        st.markdown(f"**Rating:** {rating_str} | **Relevance:** {sim_str}", unsafe_allow_html=True)
                        st.markdown("---")

        else:
            st.warning("Please enter a description first.")
    else:
        st.error("Could not load the recommendation model or data. Please check the logs.")

# --- Footer ---
st.markdown("---")
st.markdown("Movie data and posters courtesy of [TMDB](https://www.themoviedb.org/). Built using Sentence Transformers and Streamlit.")
st.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_1-5bdc75aaebeb75dc7ae79426ddd9be3b2be1e342510f8202baf6bffa71d7f5c4.svg", width=80)