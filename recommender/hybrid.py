from recommender.content_based import ContentBasedRecommender
from recommender.collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, movies_path, ratings_path):
        self.content_rec = ContentBasedRecommender(movies_path)
        self.collab_rec = CollaborativeRecommender(movies_path, ratings_path)
    
    def recommend(self, movie_title, top_n=10):
        content_recs = self.content_rec.recommend(movie_title, top_n=top_n//2)
        collab_recs = self.collab_rec.recommend(movie_title, top_n=top_n//2)
        
        recommendations = list(dict.fromkeys(content_recs + collab_recs))
        return recommendations[:top_n]
