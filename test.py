import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Création de données fictives
data = {
    "Age": [22, 25, 47, 52, 46, 56, 23, 56, 31, 40],
    "Salaire": [20000, 25000, 50000, 52000, 48000, 60000, 22000, 58000, 27000, 45000],
    "Achete": [0, 0, 1, 1, 1, 1, 0, 1, 0, 1],  # 1 = Achète, 0 = N'achète pas
}

df = pd.DataFrame(data)

X = df[["Age", "Salaire"]]
y = df["Achete"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train)
print(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")
