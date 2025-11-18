"""Genera un gráfico que resume el perfil con mayor probabilidad de suscripción."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "bank_cleaned.csv"
OUTPUT_PATH = ROOT_DIR / "reports" / "subscriber_profile.png"


def _conversion_rate(df: pd.DataFrame, column: str, top_n: int) -> pd.DataFrame:
    summary = (
        df.groupby(column, observed=False)["subscribed"]
        .mean()
        .mul(100)
        .reset_index(name="conversion_rate")
        .dropna()
        .sort_values("conversion_rate", ascending=False)
    )
    if len(summary) > top_n:
        summary = summary.head(top_n)
    return summary


def main() -> None:
    sns.set_style("whitegrid")
    df = pd.read_csv(DATA_PATH)
    if "y" not in df.columns:
        raise ValueError("La columna objetivo 'y' no se encuentra en el dataset procesado.")

    df["subscribed"] = (df["y"] == "yes").astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 120],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
        right=False,
    )
    df["previously_contacted"] = df["pdays"].apply(lambda x: "Sí" if x != 999 else "No")

    features = [
        ("age_group", "Grupo de edad"),
        ("job", "Profesión"),
        ("month", "Mes de la campaña"),
        ("contact", "Canal de contacto"),
        ("poutcome", "Resultado campaña previa"),
        ("previously_contacted", "Había sido contactado antes"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 14), sharex=False)
    axes = axes.flatten()

    for ax, (column, title) in zip(axes, features):
        top_n = 6 if column != "job" else 8
        summary = _conversion_rate(df, column, top_n)
        sns.barplot(
            data=summary,
            x="conversion_rate",
            y=column,
            hue=column,
            ax=ax,
            palette="viridis",
            legend=False,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tasa de suscripción (%)")
        ax.set_ylabel("")
        ax.set_xlim(0, max(summary["conversion_rate"].max() + 2, 10))
        for bar in ax.patches:
            width = bar.get_width()
            ax.text(
                width + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=9,
            )

    fig.suptitle(
        "Perfil de clientes con mayor probabilidad de suscripción",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Gráfico guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


