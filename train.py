"""Train a Titanic survival classifier. Run: python train.py"""
import seaborn as sns
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = sns.load_dataset("titanic")
data = df.copy()
data = data.drop(columns=["deck", "class", "who", "adult_male", "embark_town", "alive"])
data = data.dropna(subset=["embarked"])
data["age"] = data["age"].fillna(data["age"].median())
data = pd.get_dummies(data, columns=["sex", "embarked"], drop_first=True)
data["alone"] = data["alone"].astype(int)

X = data.drop(columns=["survived"])
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "model.joblib")
with open("features.json", "w") as f:
    json.dump(X.columns.tolist(), f)
print("Saved model.joblib and features.json")
