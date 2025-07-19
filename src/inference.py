import pickle
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score

with open("model_train.pkl", "rb") as f:
    model = pickle.load(f)

X, y = load_digits(return_X_y=True)
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average="macro")

print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")
