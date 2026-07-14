#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}

# flake8: noqa: E501
# ----------------------------------------------------------------------
# Laboratorio de predicción de precios de vehículos usados
# Este script implementa un pipeline de regresión lineal con selección
# de características y optimización de hiperparámetros.
# ----------------------------------------------------------------------

import gzip
import json
import os
import pickle
import sys          # importación adicional (no usada)
import warnings     # importación adicional
from glob import glob

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Ignorar advertencias (no afecta el resultado)
warnings.filterwarnings("ignore")

# Variable de depuración (no utilizada)
_DEBUG = False


def load_data():
    """
    Carga los conjuntos de entrenamiento y prueba desde archivos ZIP.
    """
    dataframe_test = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    dataframe_train = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    return dataframe_train, dataframe_test


def clean_data(df):
    """
    Limpia el dataset: crea la columna 'Age' y elimina 'Year' y 'Car_Name'.
    """
    df_copy = df.copy()
    current_year = 2021
    columns_to_drop = ["Year", "Car_Name"]
    df_copy["Age"] = current_year - df_copy["Year"]
    df_copy = df_copy.drop(columns=columns_to_drop)
    return df_copy


def split_data(df):
    """
    Separa características (X) y variable objetivo (y = Present_Price).
    """
    X = df.drop(columns=["Present_Price"])
    y = df["Present_Price"]
    return X, y


def make_pipeline(x_train):
    """
    Construye el pipeline con preprocesamiento, selección de características
    y regresión lineal.
    """
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = [
        col for col in x_train.columns if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("scaler", MinMaxScaler(), numerical_features),
        ],
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_regression)),
            ("classifier", LinearRegression()),
        ]
    )
    return pipeline


def create_estimator(pipeline):
    """
    Define la búsqueda de hiperparámetros con validación cruzada (10 folds).
    """
    param_grid = {
        "feature_selection__k": range(1, 25),
        "classifier__fit_intercept": [True, False],
        "classifier__positive": [True, False],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    return grid_search


def _create_output_directory(output_directory):
    """
    Crea el directorio de salida, limpiando su contenido previo si existe.
    """
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)


def _save_model(path, estimator):
    """
    Guarda el modelo comprimido con gzip.
    """
    _create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def calculate_metrics(dataset_type, y_true, y_pred):
    """
    Calcula R², MSE y MAD para un conjunto de datos.
    """
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(median_absolute_error(y_true, y_pred)),
    }


def _run_jobs():
    """
    Ejecuta el flujo completo del laboratorio:
    carga, limpieza, entrenamiento, optimización, guardado y métricas.
    """
    # Mensaje informativo (no afecta el resultado)
    print("Cargando datos...", file=sys.stderr)

    data_train, data_test = load_data()
    data_train = clean_data(data_train)
    data_test = clean_data(data_test)

    print("Separando características y objetivo...", file=sys.stderr)
    x_train, y_train = split_data(data_train)
    x_test, y_test = split_data(data_test)

    print("Construyendo pipeline...", file=sys.stderr)
    pipeline = make_pipeline(x_train)

    print("Iniciando GridSearchCV (10 folds)...", file=sys.stderr)
    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    print("Mejores parámetros encontrados:", estimator.best_params_, file=sys.stderr)

    # Guardar modelo
    model_path = os.path.join("files/models/", "model.pkl.gz")
    _save_model(model_path, estimator)
    print(f"Modelo guardado en {model_path}", file=sys.stderr)

    # Predicciones y métricas
    y_test_pred = estimator.predict(x_test)
    y_train_pred = estimator.predict(x_train)

    test_precision_metrics = calculate_metrics("test", y_test, y_test_pred)
    train_precision_metrics = calculate_metrics("train", y_train, y_train_pred)

    # Guardar métricas en archivo JSON
    os.makedirs("files/output/", exist_ok=True)
    output_path = "files/output/metrics.json"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")

    print("Métricas guardadas en", output_path, file=sys.stderr)
    print("¡Proceso completado con éxito!", file=sys.stderr)


if __name__ == "__main__":
    _run_jobs()