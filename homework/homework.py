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
#
import pandas as pd
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os
from glob import glob 

def load_data():

    dataframe_test = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )

    dataframe_train = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col = False,
        compression ="zip",
    )

    return dataframe_train, dataframe_test

def clean_data(df):
    df_copy = df.copy()
    current_year = 2021
    columns_to_drop = ['Year', 'Car_Name']
    df_copy["Age"] = current_year - df_copy["Year"]
    df_copy = df_copy.drop(columns=columns_to_drop)
    return df_copy

def split_data(df):
    #X , Y
    return df.drop(columns=["Present_Price"]), df["Present_Price"]

def make_pipeline(x_train):
    categorical_features=['Fuel_Type','Selling_type','Transmission']
    numerical_features= [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_features),
                ('scaler',MinMaxScaler(),numerical_features),
            ],
        )

    pipeline=Pipeline(
            [
                ("preprocessor",preprocessor),
                ('feature_selection',SelectKBest(f_regression)),
                ('classifier', LinearRegression())
            ]
        )
    return pipeline

def create_estimator(pipeline):
    param_grid = {
    'feature_selection__k':range(1,25),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]

}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True, 
        verbose= 1
      
    )

    return grid_search
    
def _create_output_directory(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)

def _save_model(path, estimator):
    _create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

def calculate_metrics(dataset_type, y_true, y_pred):
    """Calculate metrics"""
    return {
        "type": "metrics",
        "dataset": dataset_type,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred)),
    }
    

def _run_jobs():
    data_train, data_test = load_data()
    data_train = clean_data(data_train)
    data_test = clean_data(data_test)
    x_train, y_train = split_data(data_train)
    x_test, y_test = split_data(data_test)
    pipeline = make_pipeline(x_train)

    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    _save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calculate_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calculate_metrics("train", y_train, y_train_pred)


    os.makedirs("files/output/", exist_ok=True)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")

if __name__ == "__main__":
    _run_jobs()