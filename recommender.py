import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from datetime import datetime

class MovieRecommender:
    def __init__(self, data_path='movies.csv'):
        """
        Initialize the movie recommender system.
        """
        self.df = pd.read_csv(data_path)
        self.df['genres'] = self.df['genres'].apply(self.parse_genres)
        
        # Convert release date to year
        self.df['year'] = pd.to_datetime(self.df['release_date']).dt.year
        current_year = datetime.now().year
        self.df['recency_score'] = 1 + (self.df['year'] - 1970) / (current_year - 1970)
        
        # Create combined text for better matching
        self.df['combined_text'] = self.df.apply(self.create_combined_text, axis=1)
        
        # Initialize and fit TF-IDF vectorizer with custom stop words
        custom_stop_words = ['movie', 'film', 'story', 'like', 'love', 'want', 'looking', 'similar']
        self.vectorizer = TfidfVectorizer(
            stop_words=list(custom_stop_words) + ['english'],
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        
    def parse_genres(self, genres_str):
        """Parse genres from JSON string to list of genre names."""
        try:
            genres_list = json.loads(genres_str.replace("'", '"'))
            return [genre['name'] for genre in genres_list]
        except:
            return []
            
    def create_combined_text(self, row):
        """Create a combined text field for better matching."""
        # Repeat genres and emphasize certain genres
        genres = row['genres'] if isinstance(row['genres'], list) else []
        genre_weights = {
            'Science Fiction': 5,
            'Action': 4,
            'Adventure': 3,
            'Fantasy': 2,
            'Thriller': 2
        }
        
        weighted_genres = []
        for genre in genres:
            weight = genre_weights.get(genre, 1)
            weighted_genres.extend([genre.lower()] * weight)
        
        genres_text = ' '.join(weighted_genres)
        
        # Add plot with extra weight for certain keywords
        plot = row['plot']
        for keyword in ['space', 'galaxy', 'planet', 'star', 'alien', 'cosmic']:
            if keyword in plot.lower():
                plot = f"{plot} {keyword} {keyword} {keyword}"
        
        # Combine everything
        combined = f"{plot} {genres_text}"
        
        # Add keywords if available
        if 'keywords' in row and pd.notnull(row['keywords']):
            try:
                keywords = json.loads(row['keywords'].replace("'", '"'))
                keywords_text = ' '.join([k['name'] for k in keywords])
                combined += f" {keywords_text}"
            except:
                pass
                
        return combined
    
    def preprocess_query(self, query):
        """Preprocess and enhance the user query."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters but keep spaces
        query = re.sub(r'[^a-zA-Z\s]', ' ', query)
        
        # Add genre synonyms
        genre_synonyms = {
            'action': 'action adventure exciting thrilling',
            'space': 'space galaxy cosmic stellar planetary astronaut spacecraft',
            'sci-fi': 'science fiction scifi futuristic technological',
            'comedy': 'comedy funny humorous comedic',
        }
        
        enhanced_query = query
        for key, synonyms in genre_synonyms.items():
            if key in query:
                enhanced_query += f" {synonyms}"
        
        # Remove extra whitespace
        enhanced_query = ' '.join(enhanced_query.split())
        return enhanced_query
    
    def get_recommendations(self, query, n_recommendations=5):
        """Get movie recommendations based on the input query."""
        # Preprocess and enhance the query
        processed_query = self.preprocess_query(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate base similarity scores
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Apply recency boost to similarity scores
        adjusted_scores = similarity_scores * self.df['recency_score'].values
        
        # Get indices of top N similar movies
        top_indices = adjusted_scores.argsort()[-n_recommendations:][::-1]
        
        # Create list of recommendations
        recommendations = [
            (self.df.iloc[idx]['title'],
             self.df.iloc[idx]['rating'],
             self.df.iloc[idx]['num_votes'],
             similarity_scores[idx],
             ', '.join(self.df.iloc[idx]['genres']) if isinstance(self.df.iloc[idx]['genres'], list) else '',
             self.df.iloc[idx]['year'])
            for idx in top_indices
        ]
        
        return recommendations

def main():
    """Main function to demonstrate the movie recommender system."""
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Get user input
    query = input("Enter your movie preferences: ")
    if not query:
        query = "I like action movies set in space"  # Default query
    
    # Get recommendations
    recommendations = recommender.get_recommendations(query)
    
    # Print results
    print(f"\nQuery: {query}\n")
    print("Recommendations:")
    for i, (title, rating, votes, similarity, genres, year) in enumerate(recommendations, 1):
        print(f"{i}. {title} ({year})")
        print(f"   Rating: {rating}/10 ({votes:,} votes)")
        print(f"   Genres: {genres}")
        print(f"   Similarity Score: {similarity:.3f}")
        print()

if __name__ == "__main__":
    main()