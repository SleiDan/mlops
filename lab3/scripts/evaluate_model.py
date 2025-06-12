import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
