import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample movie dataset
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Inception', 'The Matrix', 'Interstellar', 'The Prestige', 'John Wick'],
    'genres': ['Sci-Fi Thriller', 'Sci-Fi Action', 'Sci-Fi Drama', 'Drama Mystery', 'Action Thriller']
})

# Sample user ratings
user_ratings = pd.DataFrame({
    'user_id': [101, 101, 102, 103],
    'movie_id': [1, 3, 2, 4],
    'rating': [5, 4, 5, 4]
})

# TF-IDF Vectorization of genres
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Get user 101's rated movies
user_id = 101
rated_movies = user_ratings[user_ratings['user_id'] == user_id]
rated_movie_ids = rated_movies['movie_id'].tolist()

# Compute weighted scores for unrated movies
scores = {}
for idx, row in movies.iterrows():
    if row['movie_id'] not in rated_movie_ids:
        sim_scores = 0
        weights = 0
        for _, rated_row in rated_movies.iterrows():
            rated_idx = movies[movies['movie_id'] == rated_row['movie_id']].index[0]
            similarity = similarity_matrix[idx][rated_idx]
            sim_scores += similarity * rated_row['rating']
            weights += similarity
        scores[row['title']] = sim_scores / weights if weights != 0 else 0

# Sort by score
recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Output
print("Movie Recommendations for User 101:")
for title, score in recommendations:
    print(f"{title}: {score:.2f}")
