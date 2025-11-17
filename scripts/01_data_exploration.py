# limpieza inicial del dataset

import pandas as pd
import numpy as np
import os

def load_data():
    #Carga el dataset desde data/raw
    data_path = os.path.join('data', 'raw', 'bank-additional-full.csv')
    df = pd.read_csv(data_path, sep=';')
    return df

def explore_data(df):
    #Explora el dataset y muestra información básica
    print("=" * 60)
    print("EXPLORACIÓN INICIAL DEL DATASET")
    print("=" * 60)
    print(f"\nForma del dataset: {df.shape}")
    print(f"\nColumnas: {list(df.columns)}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    print(f"\nValores nulos originales:\n{df.isnull().sum()}")
    print(f"\nValores 'unknown' por columna:\n{(df == 'unknown').sum()}")
    print(f"\nDistribución de la variable objetivo (y):\n{df['y'].value_counts()}")
    print(f"\nProporción de la variable objetivo:\n{df['y'].value_counts(normalize=True)}")
    print("\n" + "=" * 60)
    return df

def clean_data(df):
    #Limpia el dataset: cambia 'unknown' por NaN
    print("\nLIMPIEZA DE DATOS")
    print("=" * 60)
    
    # Crear copia para no modificar el original
    df_clean = df.copy()
    
    # Reemplazar 'unknown' por NaN en todas las columnas
    df_clean = df_clean.replace('unknown', np.nan)
    
    print(f"\nValores 'unknown' reemplazados por NaN")
    print(f"Valores nulos después de la limpieza:\n{df_clean.isnull().sum()}")
    
    # Información sobre valores únicos en columnas categóricas
    print("\nValores únicos en columnas categóricas:")
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'y':  # Excluir la variable objetivo
            print(f"\n{col}: {df_clean[col].unique()}")
    
    print("\n" + "=" * 60)
    return df_clean

def save_cleaned_data(df, filename='bank_cleaned.csv'):
    #Guarda el dataset limpio en data/processed
    output_path = os.path.join('data', 'processed', filename)
    df.to_csv(output_path, index=False)
    print(f"\nDataset limpio guardado en: {output_path}")
    return output_path

if __name__ == "__main__":
    # Cargar datos
    df = load_data()
    
    # Explorar datos
    df = explore_data(df)
    
    # Limpiar datos
    df_clean = clean_data(df)
    
    # Guardar datos limpios
    save_cleaned_data(df_clean)
    
    print("\n✓ Exploración y limpieza completadas")

