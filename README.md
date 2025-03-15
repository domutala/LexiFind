# LexiFind - API de Détection de Langue

Ce projet est une API Flask permettant de détecter la langue d'un texte en utilisant un modèle de classification basé sur TF-IDF et Naïve Bayes.

## 🚀 Installation

### Prérequis
Assurez-vous d'avoir **Python 3.x** installé sur votre machine.

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/domutala/lexifind.git
cd lexifind
```

### 2️⃣ Créer un environnement virtuel (optionnel mais recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
venv\Scripts\activate  # Sur Windows
```

### 3️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📌 Entraînement du modèle
Avant de lancer l'API, il faut entraîner le modèle et sauvegarder les fichiers nécessaires :
```bash
python train.py
```
Cela génère les fichiers **language_model.pkl** et **vectorizer.pkl**.

## 🏃 Lancer l'API
```bash
python app.py
```
L'API sera accessible sur :
```
http://127.0.0.1:5000
```

## 📡 Utilisation de l'API
### 1️⃣ Faire une requête de détection de langue
Envoyer une requête **POST** avec un texte en JSON :
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Bonjour, comment ça va ?"}'
```

### 2️⃣ Réponse JSON
```json
{
  "language": "fr"
}
```


## 📄 Fichiers Importants
- `app.py` : Code principal de l'API Flask
- `train.py` : Script d'entraînement du modèle
- `requirements.txt` : Liste des dépendances
- `model.pkl` & `vectorizer.pkl` : Modèle et vectorizer sauvegardés

## 📜 Licence
Ce projet est sous licence **MIT**.

---

