import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD

# Page configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Custom CSS for styling with pastel sage green background
st.markdown("""
<style>
    body {
        background-color: #581845; /* Pastel sage green */
        color: #FFFFFF;
    }
    .stApp {
        background-color: #581845; /* Pastel sage green */
    }
    .title {
        font-size: 3em;
        color: #FFD700;
        text-align: center;
        padding: 20px 0;
    }
    .prompt {
        font-size: 1.5em;
        color: #ADD8E6;
        text-align: center;
        padding: 10px 0;
    }
    .bubble-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        padding: 20px 0;
    }
    .movie-bubble {
        background-color: #4B0082;
        color: white;
        border: none;
        padding: 15px 25px;
        margin: 10px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 1.2em;
        transition: all 0.3s;
    }
    .movie-bubble:hover {
        background-color: #8A2BE2;
        transform: scale(1.05);
    }
    .movie-bubble-selected {
        background-color: #00CED1;
        color: #1E1E1E;
    }
    .recommend-button {
        background-color: #FFD700;
        color: #1E1E1E;
        font-size: 1.2em;
        font-weight: bold;
        padding: 15px 30px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        display: block;
        margin: 40px auto 0 auto;
        transition: all 0.3s;
    }
    .recommend-button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
    }
    .selected-movies {
        font-size: 1.4em;
        color: #ADD8E6;
        margin-bottom: 10px;
    }
    .selected-movie-item {
        color: #00008B; /* Dark blue color for selected movies */
        font-weight: bold;
        font-size: 1.3em;
    }
    .movie-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #FFD700;
    }
    .movie-genre {
        font-size: 1.2em;
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .movie-rating {
        font-size: 1.1em;
        color: #FFA500;
        font-family: 'Courier New', Courier, monospace;
    }
</style>
""", unsafe_allow_html=True)

# Title and prompt
st.markdown('<h1 class="title">🎬 Ultimate Movie Recommender 🍿</h1>', unsafe_allow_html=True)
st.markdown('<p class="prompt">Let\'s pick your favorite movies so far!</p>', unsafe_allow_html=True)

# Load the dataset (ratings.csv and movies.csv)
@st.cache_data
def load_data():
    file_path = r"C:\Users\8noor\Desktop\Seerat\Recommendation\ratings.csv"
    movies_path = r"C:\Users\8noor\Desktop\Seerat\Recommendation\movies.csv"
    ratings = pd.read_csv(file_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

ratings, movies = load_data()

# Find the 8 most popular movies (most rated)
top_movie_ids = ratings['movieId'].value_counts().head(8).index.tolist()
top_movies = movies[movies['movieId'].isin(top_movie_ids)]

# Initialize session state for selected movies
if 'selected_movies' not in st.session_state:
    st.session_state.selected_movies = []

# Display the top 8 movie titles as buttons styled as bubbles
st.markdown('<div class="bubble-container">', unsafe_allow_html=True)
for _, row in top_movies.iterrows():
    movie_title = row['title']
    is_selected = movie_title in st.session_state.selected_movies
    button_class = "movie-bubble movie-bubble-selected" if is_selected else "movie-bubble"
    if st.button(movie_title, key=movie_title, help=f"Select {movie_title}", 
                 on_click=lambda t=movie_title: st.session_state.selected_movies.append(t) 
                 if t not in st.session_state.selected_movies 
                 else st.session_state.selected_movies.remove(t)):
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Display selected movies as a list in dark blue
if st.session_state.selected_movies:
    st.markdown('<p class="selected-movies">🎥 Your selected movies:</p>', unsafe_allow_html=True)
    for movie in st.session_state.selected_movies:
        st.markdown(f'<p class="selected-movie-item">• {movie}</p>', unsafe_allow_html=True)

# Recommendation button
if st.button("Let's Find Your Next Watch That You Will Love!", key="submit_button", 
             help="Find your next favorite movie"):
    with st.spinner("🔮 Generating your personalized recommendations..."):
        if st.session_state.selected_movies:
            # Find the movie IDs for the selected movies
            selected_movie_ids = movies[movies['title'].isin(st.session_state.selected_movies)]['movieId'].values

            # Train the SVD model
            reader = Reader(rating_scale=(0.5, 5.0))
            data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
            trainset = data.build_full_trainset()
            model = SVD()
            model.fit(trainset)
            
            # Get the list of all movie IDs
            movie_ids = movies['movieId'].unique()

            # Remove the selected movie IDs from the list of movie IDs
            movies_to_predict = [movie for movie in movie_ids if movie not in selected_movie_ids]

            # Predict ratings for all movies not selected
            predictions = [model.predict(1, movie) for movie in movies_to_predict]  # Using User ID 1 as an example

            # Sort the predictions by estimated rating
            predictions.sort(key=lambda x: x.est, reverse=True)

            # Display the top 5 recommendations
            st.subheader("🌟 Top 5 Movie Recommendations Just for You:")
            for pred in predictions[:5]:
                # Get the title and genre for the movie using its movieId
                movie_title = movies[movies['movieId'] == pred.iid]['title'].values[0]
                movie_genre = movies[movies['movieId'] == pred.iid]['genres'].values[0]
                st.markdown(f"<p class='movie-title'>{movie_title}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='movie-genre'>(Genre: {movie_genre})</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='movie-rating'>Predicted Rating: {pred.est:.2f}</p>", unsafe_allow_html=True)
        else:
            st.warning("Please select at least one movie to get personalized recommendations.")
