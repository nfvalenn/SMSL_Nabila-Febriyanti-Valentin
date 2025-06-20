import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.sklearn.autolog()

mlflow.set_tracking_uri("https://dagshub.com/nfvalenn/mental-health-Nabila-Febriyanti-Valentin.mlflow")
mlflow.set_experiment("Mental Health Prediction Tuning")

df = pd.read_csv("mental_health_cleaned.csv")

X = df.drop(columns=['treatment'])
y = df['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Manual Logging Sesuai Kriteria Skilled/Advance
    mlflow.log_param("best_n_estimators", grid.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid.best_params_['min_samples_split'])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(grid.best_estimator_, "model")

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)