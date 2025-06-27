import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_path):
        self.movies = pd.read_csv(movies_path)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['genres'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def recommend(self, movie_title, top_n=5):
        indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
        idx = indices.get(movie_title)
        
        if idx is None:
            return []
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        return self.movies['title'].iloc[movie_indices].tolist()
