from flask import Flask, render_template, request
import requests
import os
import joblib
import numpy as np

# Load environment variables
TMDB_API_KEY = os.environ["58e0a31d2c3aad285afa543aea6e73bb"]

app = Flask(__name__)

# Load the trained model
model = joblib.load("movie_success_model.pkl")

# Genre to numeric mapping
genre_mapping = {
    "Action": 1,
    "Comedy": 2,
    "Drama": 3,
    "Horror": 4,
    "Sci-Fi": 5,
    "Romance": 6
}

# Fetch cast popularity from TMDB API
def get_cast_popularity(cast_name):
    url = f"https://api.themoviedb.org/3/search/person?api_key={TMDB_API_KEY}&query={cast_name}"
    response = requests.get(url).json()
    if response["results"]:
        return response["results"][0]["popularity"]
    return 5.0  # Default score if cast not found

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    budget = float(request.form["budget"])
    genre = request.form["genre"]
    cast_name = request.form["cast_name"]
    runtime = float(request.form["runtime"])

    # Get numeric genre code
    genre_code = genre_mapping.get(genre, 0)

    # Fetch cast popularity
    cast_score = get_cast_popularity(cast_name)

    # Make prediction
    input_data = np.array([[budget, genre_code, cast_score, runtime]])
    prediction = model.predict(input_data)[0]

    return render_template("result.html", prediction=prediction, cast_score=round(cast_score, 2))

if __name__ == "__main__":
    app.run(debug=True)
