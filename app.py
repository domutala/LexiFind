import joblib
import os
from flask import Flask, request, jsonify
import pandas as pd
import kagglehub as kg

app = Flask(__name__)

path = kg.dataset_download("basilb2s/language-detection")
df = pd.read_csv(os.path.join(path, "Language Detection.csv"))

# Charger le dictionnaire, modèle et le vectoriseur sauvegardés
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
language_dict = joblib.load("language_dict.pkl")


# Route pour prédire la langue
@app.route("/predict", methods=["POST"])
def predict():
    # Récupérer le texte depuis la requête
    data = request.get_json(force=True)
    text = data["text"]

    # Vectoriser le texte et faire la prédiction
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    prediction_language = language_dict.get(prediction[0], "Langue inconnue")

    # Retourner la prédiction
    return jsonify({"language": prediction_language})


if __name__ == "__main__":
    app.run(debug=True)
