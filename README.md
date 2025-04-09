# NLP Mood-Based Movie Recommender (Streamlit App)

<!-- Optional: Add a screenshot/GIF of your app here! -->
<!-- ![App Screenshot](link_to_your_screenshot.png) -->

This web application, built with Streamlit, recommends movies based on user-described moods, themes, or plots. It uses Natural Language Processing techniques to understand the semantic meaning behind the user's query and find relevant movies.

## Features

*   Recommends movies based on natural language descriptions of mood or plot.
*   Uses Sentence-Transformers (`all-MiniLM-L6-v2`) to generate embeddings for movie data and user queries.
*   Calculates cosine similarity to find the best matches.
*   Displays results in a user-friendly grid with movie posters.
*   Fetches movie posters dynamically using the TMDB API.
*   Built entirely in Python using Streamlit.

## Tech Stack

*   **Frontend:** Streamlit
*   **NLP/Embeddings:** Sentence-Transformers (Hugging Face)
*   **Similarity:** Scikit-learn
*   **Data Handling:** Pandas, NumPy
*   **API Interaction:** Requests
*   **Data Source:** TMDB 5000 Movie Dataset (via Kaggle), TMDB API

## Setup and Installation (Local)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mehran-khani/nlp-movie-recommender-streamlit.git
    cd nlp-movie-recommender-streamlit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up TMDB API Key:**
    *   Get a free API key from [The Movie Database (TMDB)](https://www.themoviedb.org/settings/api).
    *   Create a folder named `.streamlit` in the project root directory.
    *   Inside `.streamlit`, create a file named `secrets.toml`.
    *   Add your API key to `secrets.toml` like this:
        ```toml
        # .streamlit/secrets.toml
        TMDB_API_KEY = "YOUR_ACTUAL_TMDB_API_KEY_HERE"
        ```
    *   **Important:** The `.gitignore` file should prevent `secrets.toml` from being uploaded to GitHub.

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The app should open in your web browser.

## Data Files

This app relies on pre-computed embeddings and metadata:
*   `movie_embeddings.npy`: NumPy array containing sentence embeddings for movies.
*   `movie_metadata.parquet`: Pandas DataFrame containing movie metadata (ID, title, year, rating, etc.).
    *(These files were generated using the process described in the [[associated Kaggle Notebook](https://www.kaggle.com/code/mehrankhani/movie-mood-matchmaker-nlp)])*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   Movie data and posters courtesy of [The Movie Database (TMDB)](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.
*   Embeddings generated using the [Sentence-Transformers](https://www.sbert.net/) library.
