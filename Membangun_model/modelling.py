import pandas as pd
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load token dari .env (pastikan ada file .env)
load_dotenv()

mlflow.set_tracking_uri("https://dagshub.com/nfvalenn/mental-health-Nabila-Febriyanti-Valentin.mlflow")
mlflow.set_experiment("Mental Health Prediction")

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# 2️⃣ Load dataset hasil preprocessing
df = pd.read_csv("mental_health_cleaned.csv")

X = df.drop("treatment", axis=1)
y = df["treatment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)

print("Training complete")
