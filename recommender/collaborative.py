import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class CollaborativeRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        
        movie_ratings = pd.merge(self.ratings, self.movies, on='movieId')
        self.movie_user_mat = movie_ratings.pivot_table(index='title', columns='userId', values='rating').fillna(0)
        self.movie_user_sparse = csr_matrix(self.movie_user_mat.values)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.movie_user_sparse)
        
    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.movie_user_mat.index:
            return []
        
        idx = self.movie_user_mat.index.get_loc(movie_title)
        distances, indices = self.model.kneighbors(self.movie_user_mat.iloc[idx, :].values.reshape(1, -1), n_neighbors=top_n+1)
        
        recommended = [self.movie_user_mat.index[i] for i in indices.flatten() if self.movie_user_mat.index[i] != movie_title]
        
        return recommended[:top_n]
