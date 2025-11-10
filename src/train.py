from typing import Tuple
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def train_model(X, y):
    """Entrena y devuelve un modelo de regresión lineal simple."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def load_data(path: str) -> pd.DataFrame:
    """Carga un CSV desde `path` y lo devuelve como DataFrame."""
    return pd.read_csv(path)


def main(data_path: str = 'data/processed/train.csv', model_path: str = 'models/model.joblib'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No existe el archivo de datos: {data_path}")

    df = load_data(data_path)
    if df.empty:
        raise ValueError("El dataset está vacío")

    if 'target' not in df.columns:
        raise ValueError("Se espera una columna 'target' en el dataset procesado")

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")
    return model


if __name__ == '__main__':
    main()
