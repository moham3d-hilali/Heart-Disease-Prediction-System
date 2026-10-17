
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("heart.csv")

X = df[["age", "trestbps", "chol", "cp"]]
y = df["target"]

import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

import joblib

joblib.dump(model, "heart_model.pkl")
print("Model saved successfully!")



