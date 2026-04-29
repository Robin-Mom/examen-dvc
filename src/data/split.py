import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Charger les données
df = pd.read_csv("data/raw_data/raw.csv") 

# 2. Séparer features (X) et target (y)
# Exemple : la colonne cible s'appelle "target"
X = df.drop("silica_concentrate", axis=1)
y = df["silica_concentrate"]

# 3. Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour le test
    random_state=42     # reproductibilité
)

# 4. export

# Si y est une Series, on la convertit en DataFrame pour l'export
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

# Export en CSV
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train_df.to_csv("data/processed_data/y_train.csv", index=False)
y_test_df.to_csv("data/processed_data/y_test.csv", index=False)

# 5. Vérification rapide
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
