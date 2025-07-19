import json
from sklearn.linear_model import LogisticRegression
from src.train import model

def test_config_file():
    with open("config/config.json") as f:
        config = json.load(f)
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

def test_model_type():
    assert isinstance(model, LogisticRegression)

def test_model_accuracy():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    score = model.score(X, y)
    assert score > 0.8
