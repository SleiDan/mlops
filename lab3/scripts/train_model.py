import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=params["train"]["test_size"])

model = LogisticRegression(max_iter=params["train"]["max_iter"])
model.fit(X_train, y_train)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
