import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Charger les données
dataset = load_dataset("papluca/language-identification")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["labels"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["labels"]

# Transformer le texte en vecteurs TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Entraîner le modèle Naïve Bayes
model = MultinomialNB()
model.fit(X_train, train_labels)

# Prédire sur les données de test
predictions = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(test_labels, predictions)
print(f"Précision du modèle : {accuracy:.4f}")


# # Sauvegarder le modèle et le vectorizer
joblib.dump(model, "language_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
