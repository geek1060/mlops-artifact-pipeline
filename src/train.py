import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

with open("config/config.json", "r") as f:
    config = json.load(f)

X, y = load_digits(return_X_y=True)

model = LogisticRegression(
    C=config["C"],
    solver=config["solver"],
    max_iter=config["max_iter"]
)
model.fit(X, y)

with open("model_train.pkl", "wb") as f:
    pickle.dump(model, f)
