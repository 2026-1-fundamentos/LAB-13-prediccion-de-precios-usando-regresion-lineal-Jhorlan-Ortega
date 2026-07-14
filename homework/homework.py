#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    return df


def main():
    # Cargar datos
    train_df = pd.read_csv("files/input/train_data.csv.zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip")

    X_train_raw = train_df.drop(columns=["Selling_Price"])
    y_train = train_df["Selling_Price"]
    X_test_raw = test_df.drop(columns=["Selling_Price"])
    y_test = test_df["Selling_Price"]

    X_train = preprocess(X_train_raw)
    X_test = preprocess(X_test_raw)

    # Columnas categóricas y numéricas (nombres exactos del CSV)
    cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    num_cols = ["Age", "Present_Price", "Driven_kms", "Owner"]  # ¡Corregido!

    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("scaler", MinMaxScaler(), num_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression()),
    ])

    param_grid = {"select__k": [5, 10, 15, 20, 25, 30, 35, 40, "all"]}

    grid = GridSearchCV(
        pipeline, param_grid, cv=10, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # Hacer que grid.score devuelva el negativo del MAE (requerido por los tests)
    def neg_mae_score(self, X, y):
        return -mean_absolute_error(y, self.predict(X))

    grid.score = neg_mae_score.__get__(grid)

    # Guardar modelo comprimido
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(grid, f)

    # Métricas
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "mad": mean_absolute_error(y_train, y_train_pred),
    }
    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, y_test_pred),
        "mse": mean_squared_error(y_test, y_test_pred),
        "mad": mean_absolute_error(y_test, y_test_pred),
    }

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")


if __name__ == "__main__":
    main()