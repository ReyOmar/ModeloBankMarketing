# Modelo de Clasificación — Bank Marketing

## Autores
- Jostein Alexander Farfán Orjuela  
- Rey Omar Esguerra Ramírez

## Descripción
Proyecto de Ciencia de Datos para construir un modelo de clasificación que identifique qué clientes tienen mayor probabilidad de suscribirse a la campaña bancaria.

## Dataset
- Nombre: **Bank Marketing**
- Link: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data


## Objetivos

### General
Desarrollar un modelo de clasificación para identificar qué clientes tienen mayor probabilidad de suscribirse a la campaña del banco, con el fin de optimizar recursos y mejorar la efectividad del marketing.

### Específicos
- Limpiar y preparar el dataset para asegurar datos consistentes y variables listas para el modelado.
- Entrenar modelos supervisados (regresión logística con `class_weight` y Random Forest) para comparar desempeño y explicabilidad.
- Optimizar el umbral/lift según las restricciones del negocio y documentar baselines de referencia.

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                    # Datos originales
│   │   └── bank-additional-full.csv
│   └── processed/              # Datos procesados
├── models/                     # Modelos entrenados y preprocesadores
├── reports/                    # Resultados, métricas y gráficos
│   └── figures/                # Matrices, ROC e importancias en PNG
├── scripts/                    # Scripts del pipeline
│   ├── 01_data_exploration.py  # Exploración y limpieza
│   ├── 02_data_preprocessing.py # Preprocesamiento
│   ├── 03_train_model.py       # Entrenamiento del modelo Regresión Logística
│   ├── 04_train_random_forest.py # Entrenamiento Random Forest
│   └── main.py                 # Script principal - Ejecuta los scripts en orden  
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

También se pueden ejecutar los script por separado:

1. **Exploración y limpieza:**
```bash
python scripts/01_data_exploration.py
```

2. **Preprocesamiento:**
```bash
python scripts/02_data_preprocessing.py
```

3. **Entrenamiento del modelo (Regresión Logística):**
```bash
python scripts/03_train_model.py
```

4. **Entrenamiento del modelo (Random Forest):**
```bash
python scripts/04_train_random_forest.py
```

## Metodología

1. **Limpieza de datos:** Se reemplazo de `unknown` por NaN y se borró `duration` (data leakage).
2. **Feature engineering:** se generaron grupos de edad, buckets de `pdays`, indicadores de contacto previo, combinaciones de campañas (`success_ratio`, `contact_intensity`, `campaign_effort`), banderas financieras (`num_financial_products`, `has_any_debt`, `default_flag`), estacionalidad (`campaign_season`, `peak_season_contact`, `midweek_call`, `cellular_peak_combo`), etc.
3. **Preprocesamiento:** Label Encoding para variables ordinales (`education`, `month`, `day_of_week`, `poutcome`, `pdays_bucket`) y One-Hot Encoding para el resto, seguido de estandarización de todas las features numéricas.
4. **Modelado:** 
   - Regresión logística con `class_weight='balanced'` en lugar de `SMOTE` por hacerlo más simple y no duplicar datos, además de dar mayor interpretabilidad y análisis de coeficientes.
   - Random Forest con `class_weight='balanced_subsample'` para capturar interacciones no lineales y maximizar lift.
5. **Optimización de umbral:** búsqueda del umbral que maximiza F1 y cálculo adicional de umbrales que cumplen objetivos.
6. **Evaluación completa:** métricas clásicas (Accuracy, Precision, Recall, F1, ROC-AUC), reporte por clase, comparación contra baselines (mayoría/aleatorio) y análisis de lift para los percentiles superiores.


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
  - `metrics.json` / `metrics_report.txt` / `feature_importance.csv`: salidas de la regresión logística.
  - `random_forest_metrics.json` / `random_forest_metrics_report.txt` / `random_forest_feature_importance.csv`: salidas para Random Forest.
  - `figures/`: incluye `logistic_*` y `random_forest_*` (matriz, ROC, Precision-Recall, curva de ganancia acumulada, distribución de probabilidades e importancia de variables).

  Cada vez que se ejecuta `main.py` los resultados en `models/` y `reports/` se actualizan al mas reciente.