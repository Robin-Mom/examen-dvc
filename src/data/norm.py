import os 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# importer les données

X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# on supprime les dates

X_train = X_train.drop('date', axis=1)
X_test = X_test.drop('date', axis=1)

# Initialiser le scaler
scaler = MinMaxScaler()

# Fit uniquement sur le train, puis transform sur train + test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# export

X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_test_scaled_df = pd.DataFrame(X_test_scaled)

os.makedirs("data/processed_data", exist_ok=True)
X_train_scaled_df.to_csv("data/processed_data/X_train_scaled.csv")
X_test_scaled_df.to_csv("data/processed_data/X_test_scaled.csv")


