from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_reports(prices: pd.DataFrame, panel: pd.DataFrame, output: Path) -> None:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    coverage = prices.groupby("chain").agg(
        stores=("storeId", "nunique"),
        categories=("categoryId", "nunique"),
        observations=("productId", "size"),
    )
    coverage.to_csv(tables / "coverage.csv", encoding="utf-8-sig")
    panel.to_csv(tables / "daily_panel.csv", index=False, encoding="utf-8-sig")

    counts = panel.groupby(["logicalDate", "chain"]).size().unstack()
    axis = counts.plot(marker="o", ylabel="Daily panel rows", xlabel="Logical date")
    figure = axis.get_figure()
    try:
        figure.savefig(figures / "observations_by_day.png", dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
