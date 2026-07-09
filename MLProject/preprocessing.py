import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

train_df = X_train_scaled.copy()
train_df["target"] = y_train.values
test_df = X_test_scaled.copy()
test_df["target"] = y_test.values

train_df.to_csv("california_preprocessing/california_train.csv", index=False)
test_df.to_csv("california_preprocessing/california_test.csv", index=False)

print("Preprocessing done. Files saved.")
