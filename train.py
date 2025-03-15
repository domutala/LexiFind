import os
import joblib
import json
import pandas as pd
import kagglehub as kg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Charger le fichier JSON
with open("langs.json", "r", encoding="utf-8") as file:
    languages_dict = json.load(file)

# language_mapping = joblib.load("langs.json")
language_mapping = {info["name"]: code for code, info in languages_dict.items()}


ds_hf = load_dataset("papluca/language-identification")["train"]
papluca_ds = pd.DataFrame(ds_hf)
papluca_ds = papluca_ds.rename(columns={"text": "Text", "labels": "Language"})

path = kg.dataset_download("basilb2s/language-detection")
basilb2s_df = pd.read_csv(os.path.join(path, "Language Detection.csv"))
basilb2s_df["Language"] = (
    basilb2s_df["Language"].map(language_mapping).fillna("unknown")
)

local_dataset = pd.read_csv("./dataset.csv")

df = pd.concat([basilb2s_df, local_dataset], ignore_index=True)
# df.to_csv("combined_dataset.csv", index=False)

encoder = LabelEncoder()
df["Language"] = encoder.fit_transform(df["Language"])
print(df)

# Créer un dictionnaire qui mappe les valeurs encodées vers leurs langues correspondantes
language_dict = {index: language for index, language in enumerate(encoder.classes_)}

# Vectorisation du texte avec TF-IDF
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["Text"])

# Cible (Language) est déjà encodée dans 'y'
y = df["Language"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# Créer le modèle
model = LogisticRegression(class_weight="balanced")

# # Utiliser cross_val_score pour effectuer une validation croisée avec 5 plis
# scores = cross_val_score(model, x, y, cv=20, scoring="accuracy")

# # Afficher les résultats
# print(f"Scores pour chaque pli : {scores}")
# print(f"Précision moyenne : {scores.mean() * 100:.2f}%")

# # Entraîner le modèle sur l'ensemble d'entraînement
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Sauvegarder le modèle et le vectoriseur
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Sauvegarder le dictionnaire des langues
joblib.dump(language_dict, "language_dict.pkl")
