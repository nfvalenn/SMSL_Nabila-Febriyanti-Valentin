import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.sklearn.autolog()

mlflow.set_experiment("Mental Health Prediction Tuning")

url = "https://raw.githubusercontent.com/nfvalenn/Eksperimen_Nabila-Febriyanti-Valentinn/main/preprocessing/mental_health_cleaned.csv"
df = pd.read_csv(url)

X = df.drop(columns=['treatment'])
y = df['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
    acc = accuracy_score(y_test, y_pred)

    print("Best Parameters:", grid.best_params_)
    print("Accuracy after tuning:", acc)
