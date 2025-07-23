import json
import pickle
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_trained_model(config_path="config/config.json"):
    config = load_config(config_path)

    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"]
        )
    )
    model.fit(X_train, y_train)
    return model

def main():
    model = get_trained_model()
    os.makedirs("model", exist_ok=True)
    with open("model/train.pkl", "wb") as f:
        pickle.dump(model, f)
    print("model saved to model/train.pkl")

if __name__ == "__main__":
    main()