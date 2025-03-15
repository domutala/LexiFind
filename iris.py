import pandas as pd
import os
import kagglehub as kgh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Charger les données
path = kgh.dataset_download("uciml/iris")
df = pd.read_csv(os.path.join(path, "iris.csv"))

# Prétraitement des données
df.isnull().sum()  # Affiche le nombre de valeurs manquantes par colonne

encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

scaler = MinMaxScaler()
df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] = (
    scaler.fit_transform(
        df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    )
)

# Séparer les données en deux ensembles : Entraînement et Test
X = df.drop("Species", axis=1).drop("Id", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Créer le modèle
model = LogisticRegression()

# Effectuer la validation croisée avec 5 plis (folds)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
# Afficher les scores pour chaque pli
print(f"Scores pour chaque pli : {scores}")
# Calculer la précision moyenne
print(f"Précision moyenne : {scores.mean() * 100:.2f}%")


model.fit(X_train, y_train)

# entrainement du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")
