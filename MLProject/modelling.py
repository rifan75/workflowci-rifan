import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Load preprocessed data
train_df = pd.read_csv("iris_preprocessing/iris_train.csv")
test_df = pd.read_csv("iris_preprocessing/iris_test.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

mlflow.sklearn.autolog()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_score", f1)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
