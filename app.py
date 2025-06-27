from flask import Flask, request, jsonify
from recommender.hybrid import HybridRecommender

app = Flask(__name__)

rec_sys = HybridRecommender('data/movies.csv', 'data/ratings.csv')

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('movie')
    if not movie_title:
        return jsonify({"error": "Movie title is required"}), 400
    
    recommendations = rec_sys.recommend(movie_title)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
