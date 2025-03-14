from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Charger le modèle et le vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)
CORS(app)


# Fonction pour prédire la langue
def predict_language(text):
    X_input = vectorizer.transform([text])  # Transformer le texte en vecteur TF-IDF
    prediction = model.predict(X_input)  # Faire la prédiction
    return prediction[0]  # Retourner la langue prédite


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    predicted_language = predict_language(text)
    return jsonify({"language": predicted_language})


if __name__ == "__main__":
    app.run(debug=True)
