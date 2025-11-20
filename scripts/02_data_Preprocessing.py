# preprocesamiento y ingeneiria de características

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pickle

def load_cleaned_data(filename='bank_cleaned.csv'):
    #Carga el dataset limpio
    data_path = os.path.join('data', 'processed', filename)
    df = pd.read_csv(data_path)
    
    # ELIMINAR variable 'duration' - Data Leakage
    if 'duration' in df.columns:
        print("\n ELIMINANDO variable 'duration' (data leakage)")
        df = df.drop('duration', axis=1)
    
    return df

def handle_missing_values(df):
    #Maneja valores faltantes
    print("\nMANEJO DE VALORES FALTANTES")
    print("=" * 60)
    
    df_processed = df.copy()
    
    # Para columnas categóricas, usar moda
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'y']
    
    for col in categorical_cols:
        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'missing'
        df_processed[col] = df_processed[col].fillna(mode_value)
        print(f"{col}: {df[col].isnull().sum()} valores faltantes -> rellenados con moda: {mode_value}")
    
    # Para columnas numéricas, usar mediana
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
            print(f"{col}: {df[col].isnull().sum()} valores faltantes -> rellenados con mediana: {median_value}")
    
    print(f"\nValores nulos restantes: {df_processed.isnull().sum().sum()}")
    return df_processed

def encode_categorical_variables(df):
    #Codifica variables categóricas
    print("\nCODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    print("=" * 60)
    
    df_encoded = df.copy()
    
    # Separar variables categóricas (excluyendo la variable objetivo)
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != 'y']
    
    # Usar Label Encoding para variables categoricas ordinales
    # y One-Hot Encoding para variables nominales
    label_encoders = {}
    
    # Variables que pueden ser ordinales (con orden logico)
    ordinal_vars = ['education', 'month', 'day_of_week', 'poutcome', 'pdays_bucket']
    
    # Variables nominales que necesitan one-hot encoding
    nominal_vars = [col for col in categorical_cols if col not in ordinal_vars]
    
    # Label encoding para ordinales
    for col in ordinal_vars:
        if col in categorical_cols and col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"{col}: Label Encoded")
    
    # One-hot encoding para nominales
    for col in nominal_vars:
        if col in categorical_cols and col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
            print(f"{col}: One-Hot Encoded -> {len(dummies.columns)} columnas")
    
    # Codificar variable objetivo (y: yes/no -> 1/0)
    df_encoded['y'] = (df_encoded['y'] == 'yes').astype(int)
    
    print(f"\nShape después de encoding: {df_encoded.shape}")
    return df_encoded, label_encoders

def feature_engineering(df):
    #Realiza feature engineering
    print("\nFEATURE ENGINEERING")
    print("=" * 60)
    
    df_fe = df.copy()
    
    engineered = []
    
    if 'age' in df_fe.columns:
        df_fe['age_group'] = pd.cut(
            df_fe['age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['<30', '30-40', '40-50', '50-60', '60+']
        )
        df_fe['age_group'] = df_fe['age_group'].cat.codes
        engineered.append('age_group')
    
    if 'pdays' in df_fe.columns:
        df_fe['previously_contacted'] = (df_fe['pdays'] != 999).astype(int)
        pdays_clean = df_fe['pdays'].replace(999, np.nan)
        df_fe['pdays_bucket'] = pd.cut(
            pdays_clean,
            bins=[-np.inf, 7, 30, 90, 1000],
            labels=['<1w', '1-4w', '1-3m', '>=3m']
        )
        df_fe['pdays_bucket'] = df_fe['pdays_bucket'].cat.add_categories('no_contact').fillna('no_contact')
        engineered.extend(['previously_contacted', 'pdays_bucket'])
    
    if {'previous', 'campaign'}.issubset(df_fe.columns):
        df_fe['success_ratio'] = df_fe['previous'] / (df_fe['campaign'] + 1)
        df_fe['contact_intensity'] = df_fe['campaign'] + df_fe['previous']
        df_fe['campaign_effort'] = df_fe['campaign'] / (df_fe['previous'] + 1)
        engineered.extend(['success_ratio', 'contact_intensity', 'campaign_effort'])
    
    if {'housing', 'loan'}.issubset(df_fe.columns):
        housing_flag = (df_fe['housing'] == 'yes').astype(int)
        loan_flag = (df_fe['loan'] == 'yes').astype(int)
        df_fe['num_financial_products'] = housing_flag + loan_flag
        df_fe['has_any_debt'] = ((df_fe['housing'] == 'yes') | (df_fe['loan'] == 'yes')).astype(int)
        engineered.extend(['num_financial_products', 'has_any_debt'])
    
    if 'default' in df_fe.columns:
        df_fe['default_flag'] = (df_fe['default'] == 'yes').astype(int)
        engineered.append('default_flag')
    
    if 'month' in df_fe.columns:
        season_map = {
            'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
            'mar': 'spring', 'apr': 'spring', 'may': 'spring',
            'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
            'sep': 'autumn', 'oct': 'autumn', 'nov': 'autumn'
        }
        df_fe['campaign_season'] = df_fe['month'].map(season_map).fillna('unknown')
        peak_months = {'mar', 'apr', 'sep', 'dec'}
        df_fe['peak_season_contact'] = df_fe['month'].isin(peak_months).astype(int)
        engineered.extend(['campaign_season', 'peak_season_contact'])
    
    if 'day_of_week' in df_fe.columns:
        df_fe['midweek_call'] = df_fe['day_of_week'].isin(['tue', 'wed', 'thu']).astype(int)
        engineered.append('midweek_call')
    
    if {'contact', 'month'}.issubset(df_fe.columns):
        df_fe['cellular_peak_combo'] = (
            (df_fe['contact'] == 'cellular') & df_fe['month'].isin(['mar', 'apr', 'sep', 'dec'])
        ).astype(int)
        engineered.append('cellular_peak_combo')
    
    print("Features creadas:")
    for feat in engineered:
        print(f"- {feat}")
    
    return df_fe

def prepare_train_test_split(df, test_size=0.2, random_state=42):
    #Prepara train/test split
    print("\nPREPARACIÓN DE TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Separar features y target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} muestras")
    print(f"Test set: {X_test.shape[0]} muestras")
    print(f"Proporción de clase positiva en train: {y_train.mean():.3f}")
    print(f"Proporción de clase positiva en test: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    #Estandariza las features numéricas
    print("\nESTANDARIZACIÓN DE FEATURES")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("Features estandarizadas usando StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler, label_encoders):
    #Guarda los datos procesados y los preprocesadores
    print("\nGUARDANDO DATOS PROCESADOS")
    print("=" * 60)
    
    # Guardar datasets
    X_train.to_csv(os.path.join('data', 'processed', 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join('data', 'processed', 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join('data', 'processed', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join('data', 'processed', 'y_test.csv'), index=False)
    
    # Guardar preprocesadores
    with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join('models', 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print("✓ Datos procesados guardados en data/processed/")
    print("✓ Preprocesadores guardados en models/")

if __name__ == "__main__":
    # Cargar datos limpios
    df = load_cleaned_data()
    
    # Feature engineering (antes de manejar missing values para evitar problemas)
    df = feature_engineering(df)
    
    # Manejar valores faltantes
    df = handle_missing_values(df)
    
    # Codificar variables categóricas
    df, label_encoders = encode_categorical_variables(df)
    
    # Preparar train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Estandarizar features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Guardar datos procesados
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders)
    
    print("\n✓ Preprocesamiento completado")