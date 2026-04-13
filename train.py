from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

artifact = {
    "model": model,
    "target_names": iris.target_names.tolist()
}

joblib.dump(artifact, "model.joblib")

print("Model trained successfully")
print("Accuracy:", round(acc, 4))
print("Saved as model.joblib")