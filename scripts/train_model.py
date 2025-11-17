#Entrenamiento y evaluación de la Regresión Logística con SMOTE+Tomek.

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"):
    try:
        plt.style.use(style)
        break
    except OSError:
        continue
sns.set_palette("husl")


def _class_distribution(values: pd.Series | np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    #Carga los conjuntos procesados desde disco.
    print("\nCargando datos procesados...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def balance_data(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    #Balancea las clases usando SMOTE+Tomek.
    print("\nAplicando SMOTE+Tomek para balancear clases...")
    print(f"Distribución original: {_class_distribution(y)}")
    sampler = SMOTETomek(random_state=random_state)
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    print(f"Distribución balanceada: {_class_distribution(y_balanced)}")
    return X_balanced, y_balanced


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> LogisticRegression:
    #Entrena la Regresión Logística sobre los datos balanceados.
    X_balanced, y_balanced = balance_data(X_train, y_train, random_state)
    model = LogisticRegression(
        random_state=random_state,
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
    )
    model.fit(X_balanced, y_balanced)
    print("✓ Modelo entrenado")
    return model


def find_optimal_threshold(
    y_true: pd.Series, y_scores: np.ndarray
) -> Tuple[float, float]:
    #Encuentra el umbral que maximiza el F1-score.
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if thresholds.size == 0:
        default_pred = (y_scores >= 0.5).astype(int)
        return 0.5, f1_score(y_true, default_pred)

    f1_scores = 2 * precision[1:] * recall[1:] / np.clip(
        precision[1:] + recall[1:], 1e-12, None
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    #Evalúa el modelo con umbral optimizado.
    print("\nEvaluando modelo...")
    y_train_scores = model.predict_proba(X_train)[:, 1]
    y_test_scores = model.predict_proba(X_test)[:, 1]

    optimal_threshold, train_f1 = find_optimal_threshold(y_train, y_train_scores)
    print(f"Umbral óptimo: {optimal_threshold:.3f} (F1 train: {train_f1:.4f})")

    y_test_pred = (y_test_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1": float(f1_score(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_scores)),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold": float(optimal_threshold),
        "train_f1_at_optimal_threshold": float(train_f1),
    }

    print("\n--- MÉTRICAS EN TEST ---")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key.capitalize():<10}: {metrics[key]:.4f}")
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(cm)

    return metrics, y_test_pred, y_test_scores


def get_feature_importance(
    model: LogisticRegression, feature_names: pd.Index
) -> pd.DataFrame:
    #Retorna la importancia (coeficientes) de cada feature.
    coefficients = model.coef_[0]
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 variables más importantes:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    #Genera el gráfico de matriz de confusión.
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Valor Real", fontsize=12)
    ax.set_title("Matriz de Confusión", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Matriz de confusión guardada en {save_path}")


def plot_roc_curve_chart(
    y_test: pd.Series, y_scores: np.ndarray, roc_auc: float, save_path: Path
) -> None:
    #Genera la curva ROC.
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label="Regresión Logística")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador Aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title("Curva ROC", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.text(
        0.6,
        0.2,
        f"AUC = {roc_auc:.4f}",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Curva ROC guardada en {save_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame, save_path: Path, top_n: int = 20
) -> None:
    #Genera el gráfico con los coeficientes más relevantes.
    top_features = importance_df.head(min(top_n, len(importance_df)))
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["red" if val < 0 else "green" for val in top_features["coefficient"]]
    ax.barh(top_features["feature"], top_features["coefficient"], color=colors)
    ax.set_xlabel("Coeficiente", fontsize=12)
    ax.set_title(
        f"Top {len(top_features)} Variables Más Importantes\n(Regresión Logística)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Importancia de variables guardada en {save_path}")


def save_text_report(
    metrics: Dict[str, float], importance_df: pd.DataFrame, save_path: Path
) -> None:
    #Genera un reporte plano con las métricas principales.
    cm = np.array(metrics["confusion_matrix"])
    lines = [
        "=" * 60,
        "REPORTE DE RESULTADOS - REGRESIÓN LOGÍSTICA",
        "Método: SMOTE+Tomek + Optimización de Umbral",
        "=" * 60,
        "",
        "MÉTRICAS DEL MODELO",
        "-" * 60,
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-Score:  {metrics['f1']:.4f}",
        f"ROC-AUC:   {metrics['roc_auc']:.4f}",
        f"Umbral óptimo: {metrics['optimal_threshold']:.3f}",
        "",
        "MATRIZ DE CONFUSIÓN",
        "-" * 60,
        f"Verdaderos Negativos: {cm[0, 0]}",
        f"Falsos Positivos:     {cm[0, 1]}",
        f"Falsos Negativos:     {cm[1, 0]}",
        f"Verdaderos Positivos: {cm[1, 1]}",
        "",
        "TOP 15 VARIABLES MÁS IMPORTANTES",
        "-" * 60,
    ]

    top_rows = importance_df.head(15)
    for row in top_rows.itertuples(index=False):
        lines.append(f"{row.feature:30s} | Coeficiente: {row.coefficient:8.4f}")

    lines.extend(
        [
            "",
            "=" * 60,
            "CONCLUSIONES",
            "-" * 60,
            "1. El modelo mantiene un balance entre precisión y recall tras balancear clases.",
            "2. Los coeficientes positivos indican variables que incrementan la probabilidad de suscripción.",
            "3. Puede utilizarse para priorizar clientes con mayor probabilidad de respuesta.",
            "",
        ]
    )

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Reporte guardado en {save_path}")


def export_artifacts(
    model: LogisticRegression,
    metrics: Dict[str, float],
    importance_df: pd.DataFrame,
    y_test: pd.Series,
    y_test_scores: np.ndarray,
) -> None:
    #Persiste el modelo, métricas y visualizaciones.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Modelo guardado en {MODELS_DIR / 'model.pkl'}")

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print(f"✓ Métricas e importancia guardadas en {REPORTS_DIR}")

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curve_chart(y_test, y_test_scores, metrics["roc_auc"], REPORTS_DIR / "roc_curve.png")
    plot_feature_importance(importance_df, REPORTS_DIR / "feature_importance.png")
    save_text_report(metrics, importance_df, REPORTS_DIR / "metrics_report.txt")


def main():
    print("=" * 60)
    print("ENTRENAMIENTO - REGRESIÓN LOGÍSTICA (SMOTE+Tomek)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_processed_data()
    model = train_logistic_regression(X_train, y_train)
    metrics, _, y_test_scores = evaluate_model(model, X_train, X_test, y_train, y_test)
    importance_df = get_feature_importance(model, X_train.columns)
    export_artifacts(model, metrics, importance_df, y_test, y_test_scores)

    print("\n✓ Proceso completado. Resultados disponibles en 'reports/'.")


if __name__ == "_main_":
    main()