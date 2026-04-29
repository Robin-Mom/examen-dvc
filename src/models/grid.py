from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

# 1. import des données
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv") 
y_train = pd.read_csv("data/processed_data/y_train.csv" )
y_train = y_train.values.ravel()

# 2. Modèle
model = RandomForestRegressor(random_state=42)

# 3. Grille d'hyperparamètres
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

# 4. GridSearch
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",  # métrique pour régression
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 5. Entraînement
grid.fit(X_train_scaled, y_train)

# 6. Résultats
print("Meilleurs paramètres :", grid.best_params_)
print("Meilleur score (MSE négatif) :", grid.best_score_)

# 7. export

joblib.dump(grid.best_params_, "models/best_params.pkl")
