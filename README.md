# Modelo de Clasificación — Bank Marketing

## Autores
- Jostein Alexander Farfán Orjuela  
- Rey Omar Esguerra Ramírez

## Descripción
Proyecto de Ciencia de Datos para construir un modelo de clasificación que identifique qué clientes tienen mayor probabilidad de suscribirse a la campaña bancaria.

## Dataset
- Nombre: **Bank Marketing**
- Link: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data
- Objetivo: Predecir la suscripción de clientes a la campaña

## Objetivos

### General
Desarrollar un modelo de clasificación para identificar qué clientes tienen mayor probabilidad de suscribirse a la campaña del banco, con el fin de optimizar recursos y mejorar la efectividad del marketing.

### Específicos
- Limpiar y preparar el dataset para asegurar datos consistentes y variables listas para el modelado.
- Entrenar un modelo de regresión logística con balanceo de clases usando SMOTE+Tomek.
- Optimizar el umbral de decisión para maximizar el F1-score.

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                    # Datos originales
│   │   └── bank-additional-full.csv
│   └── processed/              # Datos procesados
├── models/                     # Modelos entrenados y preprocesadores
├── reports/                    # Resultados, métricas y gráficos
├── scripts/                    # Scripts del pipeline
│   ├── 01_data_exploration.py  # Exploración y limpieza
│   ├── 02_data_preprocessing.py # Preprocesamiento
│   ├── train_model.py          # Entrenamiento del modelo
│   └── main.py                 # Script principal
└── requirements.txt            # Dependencias
```

## Instalación

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar el pipeline completo

Para ejecutar todo el pipeline de principio a fin:

```bash
python scripts/main.py
```

### Ejecutar scripts individuales

También puedes ejecutar cada script por separado:

1. **Exploración y limpieza:**
```bash
python scripts/01_data_exploration.py
```

2. **Preprocesamiento:**
```bash
python scripts/02_data_preprocessing.py
```

3. **Entrenamiento del modelo:**
```bash
python scripts/train_model.py
```

## Metodología

1. **Limpieza de datos**: Reemplazo de valores "unknown" por NaN y manejo de valores faltantes
2. **Feature Engineering**: Creación de nuevas variables (grupos de edad, ratios, etc.)
3. **Preprocesamiento**: Codificación de variables categóricas (Label Encoding y One-Hot Encoding) y estandarización
4. **Balanceo de clases**: SMOTE + Tomek Links para manejar el desbalance del dataset
5. **Modelado**: Regresión Logística
6. **Optimización de umbral**: Búsqueda del umbral óptimo que maximiza el F1-score
7. **Evaluación**: Métricas de clasificación (Accuracy, Precision, Recall, F1, ROC-AUC)

## Método Recomendado

El modelo utiliza **SMOTE+Tomek con optimización de umbral**:

- **SMOTE (Synthetic Minority Oversampling Technique)**: Genera muestras sintéticas de la clase minoritaria para balancear las clases.
- **Tomek Links**: Elimina muestras ruidosas y mejora la separación entre clases.
- **Optimización de umbral**: Encuentra el umbral de decisión óptimo que maximiza el F1-score, mejorando el balance entre Precision y Recall.

Este método ha demostrado mejores resultados que el modelo original, especialmente en Precision, que es crítica para campañas de marketing bancario.

## Resultados

Después de ejecutar el pipeline, se encontraran:

- **Datos procesados** en `data/processed/`:
  - `bank_cleaned.csv`: Dataset limpio (unknown → NaN)
  - `X_train.csv`, `X_test.csv`: Features de entrenamiento y prueba
  - `y_train.csv`, `y_test.csv`: Variables objetivo

- **Modelos** en `models/`:
  - `model.pkl`: Modelo entrenado
  - `scaler.pkl`: Estandarizador de features
  - `label_encoders.pkl`: Codificadores de variables categóricas

- **Resultados y reportes** en `reports/`:
  - `metrics.json`: Métricas del modelo en formato JSON
  - `metrics_report.txt`: Reporte completo de métricas en texto
  - `feature_importance.csv`: Importancia de variables
  - `confusion_matrix.png`: Matriz de confusión
  - `roc_curve.png`: Curva ROC
  - `feature_importance.png`: Gráfico de importancia de variables