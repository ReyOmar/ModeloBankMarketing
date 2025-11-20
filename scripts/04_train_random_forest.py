"""Entrena y evalúa un Random Forest para Bank Marketing."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
FIGURES_DIR = REPORTS_DIR / "figures"

for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"):
    try:
        plt.style.use(style)
        break
    except OSError:
        continue
sns.set_palette("husl")


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("\nCargando datos procesados...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestClassifier:
    print("\nEntrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("✓ Random Forest entrenado")
    return model


def find_optimal_threshold(y_true: pd.Series, y_scores: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if thresholds.size == 0:
        default_pred = (y_scores >= 0.5).astype(int)
        return 0.5, f1_score(y_true, default_pred)

    f1_scores = 2 * precision[1:] * recall[1:] / np.clip(
        precision[1:] + recall[1:], a_min=1e-12, a_max=None
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def search_threshold_for_target(
    y_true: pd.Series, y_scores: np.ndarray, target: float
) -> Dict[str, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    recall_threshold = next(
        (
            (float(thresholds[i]), float(precision[i + 1]), float(recall[i + 1]))
            for i in range(len(thresholds))
            if recall[i + 1] >= target
        ),
        (None, None, None),
    )

    precision_threshold = next(
        (
            (float(thresholds[i]), float(precision[i + 1]), float(recall[i + 1]))
            for i in range(len(thresholds))
            if precision[i + 1] >= target
        ),
        (None, None, None),
    )

    return {
        "target": target,
        "recall_threshold": recall_threshold[0],
        "recall_threshold_precision": recall_threshold[1],
        "recall_threshold_recall": recall_threshold[2],
        "precision_threshold": precision_threshold[0],
        "precision_threshold_precision": precision_threshold[1],
        "precision_threshold_recall": precision_threshold[2],
    }


def _summarize_binary_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def compute_baseline_metrics(
    y_test: pd.Series, y_train: pd.Series, random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    positive_rate = float(y_train.mean())
    majority_pred = np.zeros_like(y_test, dtype=int)
    rng = np.random.default_rng(random_state)
    random_pred = rng.binomial(1, positive_rate, size=len(y_test))

    return {
        "majority": _summarize_binary_metrics(y_test, majority_pred),
        "random": _summarize_binary_metrics(y_test, random_pred),
        "positive_rate": positive_rate,
    }


def compute_lift_metrics(
    y_true: pd.Series, scores: np.ndarray, fractions: Tuple[float, ...] = (0.1, 0.2)
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    base_rate = float(y_true.mean())
    order = np.argsort(scores)[::-1]
    y_ordered = y_true.iloc[order].reset_index(drop=True)

    for frac in fractions:
        cutoff = max(1, int(len(y_true) * frac))
        top_rate = float(y_ordered.iloc[:cutoff].mean())
        lift = top_rate / base_rate if base_rate > 0 else float("nan")
        key = f"top_{int(frac * 100)}pct"
        results[key] = {
            "rate": top_rate,
            "lift": lift,
            "cutoff": cutoff,
        }

    results["base_rate"] = base_rate
    return results


def evaluate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    print("\nEvaluando modelo RF...")
    y_train_scores = model.predict_proba(X_train)[:, 1]
    y_test_scores = model.predict_proba(X_test)[:, 1]

    optimal_threshold, train_f1 = find_optimal_threshold(y_train, y_train_scores)
    print(f"Umbral óptimo: {optimal_threshold:.3f} (F1 train: {train_f1:.4f})")

    y_test_pred = (y_test_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
        zero_division=0,
        target_names=["no", "yes"],
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1": float(f1_score(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_scores)),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "optimal_threshold": float(optimal_threshold),
        "train_f1_at_optimal_threshold": float(train_f1),
        "baselines": compute_baseline_metrics(y_test, y_train),
        "lift": compute_lift_metrics(y_test, y_test_scores),
        "target_thresholds": search_threshold_for_target(y_test, y_test_scores, 0.65),
    }

    print("\n--- MÉTRICAS EN TEST (RF) ---")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{key.capitalize():<10}: {metrics[key]:.4f}")
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(cm)
    return metrics, y_test_pred, y_test_scores


def extract_feature_importance(
    model: RandomForestClassifier, feature_names: pd.Index
) -> pd.DataFrame:
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Valor Real", fontsize=12)
    ax.set_title("Random Forest - Matriz de Confusión", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Matriz de confusión guardada en {save_path}")


def plot_roc_curve_chart(
    y_test: pd.Series, y_scores: np.ndarray, roc_auc: float, save_path: Path
) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador Aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title("Random Forest - Curva ROC", fontsize=14, fontweight="bold")
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
    top_features = importance_df.head(min(top_n, len(importance_df)))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_features["feature"], top_features["importance"], color="forestgreen")
    ax.set_xlabel("Importancia (Gini)", fontsize=12)
    ax.set_title(
        f"Top {len(top_features)} Importancias — Random Forest", fontsize=14, fontweight="bold"
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Importancia de variables guardada en {save_path}")


def plot_precision_recall_curve_chart(
    y_test: pd.Series, y_scores: np.ndarray, save_path: Path
) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="purple", linewidth=2, label="Random Forest")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Curva Precision-Recall (Random Forest)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Precision-Recall guardada en {save_path}")


def plot_cumulative_gain_chart(
    y_test: pd.Series, y_scores: np.ndarray, save_path: Path
) -> None:
    order = np.argsort(y_scores)[::-1]
    y_sorted = y_test.iloc[order].reset_index(drop=True)
    cumulative_positives = y_sorted.cumsum()
    total_positives = cumulative_positives.iloc[-1]
    samples = np.arange(1, len(y_test) + 1)
    gain = cumulative_positives / total_positives
    percentage_samples = samples / len(y_test)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(percentage_samples, gain, label="Modelo", color="darkgreen", linewidth=2)
    ax.plot(percentage_samples, percentage_samples, "--", color="gray", label="Aleatorio")
    ax.set_xlabel("% Clientes contactados", fontsize=12)
    ax.set_ylabel("% Suscripciones capturadas", fontsize=12)
    ax.set_title("Curva de Ganancia Acumulada (Random Forest)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Gain chart guardado en {save_path}")


def plot_score_distribution(
    y_test: pd.Series, y_scores: np.ndarray, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        y_scores[y_test == 1],
        bins=30,
        color="seagreen",
        label="Suscribió",
        ax=ax,
        alpha=0.6,
    )
    sns.histplot(
        y_scores[y_test == 0],
        bins=30,
        color="salmon",
        label="No suscribió",
        ax=ax,
        alpha=0.6,
    )
    ax.set_xlabel("Probabilidad estimada", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Distribución de probabilidades (Random Forest)", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Distribución de scores guardada en {save_path}")


def save_text_report(metrics: Dict, importance_df: pd.DataFrame, save_path: Path) -> None:
    cm = np.array(metrics["confusion_matrix"])
    class_report = metrics.get("classification_report", {})
    baselines = metrics.get("baselines", {})
    lift_info = metrics.get("lift", {})
    target_info = metrics.get("target_thresholds", {})

    lines = [
        "=" * 60,
        "REPORTE DE RESULTADOS - RANDOM FOREST",
        "Método: class_weight balanced_subsample + Optimización de Umbral",
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
        "REPORTE POR CLASE (Precision / Recall / F1 / Soporte)",
        "-" * 60,
        f"{'Clase':<12} | Precision | Recall |    F1 | Soporte",
        "-" * 60,
    ]

    ordered_keys = ["no", "yes"]
    for key in ordered_keys:
        if key in class_report:
            stats = class_report[key]
            lines.append(
                f"{key.upper():<12} | {stats['precision']:>9.4f} | "
                f"{stats['recall']:>6.4f} | {stats['f1-score']:>6.4f} | "
                f"{int(stats['support']):>7}"
            )

    for label in ("macro avg", "weighted avg"):
        if label in class_report:
            stats = class_report[label]
            display = label.title()
            lines.append(
                f"{display:<12} | {stats['precision']:>9.4f} | "
                f"{stats['recall']:>6.4f} | {stats['f1-score']:>6.4f} | "
                f"{'-':>7}"
            )

    if target_info:
        lines.extend(
            [
                "",
                "UMBRAL PARA OBJETIVO 65% PRECISIÓN/RECALL",
                "-" * 60,
                f"Target solicitado: {target_info.get('target', 0.65):.2f}",
                f"Recall >= target: umbral={target_info.get('recall_threshold')}, "
                f"precision={target_info.get('recall_threshold_precision')}, "
                f"recall={target_info.get('recall_threshold_recall')}",
                f"Precision >= target: umbral={target_info.get('precision_threshold')}, "
                f"precision={target_info.get('precision_threshold_precision')}, "
                f"recall={target_info.get('precision_threshold_recall')}",
            ]
        )

    lines.extend(
        [
            "",
            "TOP 15 VARIABLES MÁS IMPORTANTES",
            "-" * 60,
        ]
    )
    top_rows = importance_df.head(15)
    for row in top_rows.itertuples(index=False):
        lines.append(f"{row.feature:30s} | Importancia: {row.importance:8.4f}")

    lines.extend(
        [
            "",
            "=" * 60,
        ]
    )
    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Reporte guardado en {save_path}")


def export_artifacts(
    model: RandomForestClassifier,
    metrics: Dict,
    importance_df: pd.DataFrame,
    y_test: pd.Series,
    y_test_scores: np.ndarray,
    prefix: str = "random_forest",
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{prefix}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Modelo guardado en {model_path}")

    metrics_path = REPORTS_DIR / f"{prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    importance_df.to_csv(REPORTS_DIR / f"{prefix}_feature_importance.csv", index=False)

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, FIGURES_DIR / f"{prefix}_confusion_matrix.png")
    plot_roc_curve_chart(
        y_test, y_test_scores, metrics["roc_auc"], FIGURES_DIR / f"{prefix}_roc_curve.png"
    )
    plot_feature_importance(
        importance_df, FIGURES_DIR / f"{prefix}_feature_importance.png"
    )
    plot_precision_recall_curve_chart(
        y_test, y_test_scores, FIGURES_DIR / f"{prefix}_precision_recall.png"
    )
    plot_cumulative_gain_chart(
        y_test, y_test_scores, FIGURES_DIR / f"{prefix}_cumulative_gain.png"
    )
    plot_score_distribution(
        y_test, y_test_scores, FIGURES_DIR / f"{prefix}_score_distribution.png"
    )
    save_text_report(metrics, importance_df, REPORTS_DIR / f"{prefix}_metrics_report.txt")


def main():
    print("=" * 60)
    print("ENTRENAMIENTO - RANDOM FOREST")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_processed_data()
    model = train_random_forest(X_train, y_train)
    metrics, _, y_test_scores = evaluate_model(model, X_train, X_test, y_train, y_test)
    importance_df = extract_feature_importance(model, X_train.columns)
    export_artifacts(model, metrics, importance_df, y_test, y_test_scores)
    print("\n✓ Random Forest completado. Resultados en 'reports/'")


if __name__ == "__main__":
    main()

