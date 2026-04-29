import joblib
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# chargement 

model = joblib.load("models/RF.pkl")
X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv") 
y_test = pd.read_csv("data/processed_data/y_test.csv" )
y_test = y_test.values.ravel()

# metrics

y_pred = model.predict(X_test_scaled)

print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))

# export

scores = {
    "mse": mean_squared_error(y_test, y_pred),
    "rmse": mean_squared_error(y_test, y_pred),
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("data/processed_data/predicted.csv")
