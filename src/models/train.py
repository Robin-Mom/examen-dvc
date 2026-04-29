import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Chargement
best_params = joblib.load("models/best_params/best_params.pkl")
X_train_scaled = pd.read_csv("data/normalized_data/X_train_scaled.csv") 
y_train = pd.read_csv("data/processed_data/y_train.csv" )
y_train = y_train.values.ravel()

# Recréer le modèle
model = RandomForestRegressor(**best_params, random_state=42)

# Refit
model.fit(X_train_scaled, y_train)

# export

os.makedirs("models/best_models", exist_ok=True)
joblib.dump(model, "models/best_models/RF.pkl")

