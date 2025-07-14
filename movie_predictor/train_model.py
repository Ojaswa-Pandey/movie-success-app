import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample dataset (replace with real TMDB dataset)
data = {
    "budget": [100, 20, 150, 10, 200, 30],
    "genre": [1, 2, 3, 4, 1, 2],
    "cast_score": [9.0, 3.0, 8.5, 4.5, 9.5, 2.5],
    "runtime": [120, 90, 150, 80, 160, 100],
    "success": ["Hit", "Flop", "Hit", "Flop", "Hit", "Flop"]
}

df = pd.DataFrame(data)

# Encode target labels
df["success"] = df["success"].map({"Flop": 0, "Hit": 1})

# Split data
X = df[["budget", "genre", "cast_score", "runtime"]]
y = df["success"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "movie_success_model.pkl")
print("Model saved as movie_success_model.pkl")
