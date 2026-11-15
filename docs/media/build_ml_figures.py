"""Replot two documentation figures from frozen, committed evaluation records.

This script reads saved results only. It never loads a model or source dataset.
Run from any directory with: python docs/media/build_ml_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RELEASE = ROOT / "reports/release_model_evaluation/evaluation.json"
CONFUSION = (
    ROOT
    / "reports/candidate_final_evaluation/recorded_results/evaluation/confusion_counts.csv"
)


def matched_v1_v3() -> None:
    saved = json.loads(RELEASE.read_text(encoding="utf-8"))
    views = [
        ("up_to_1", "Up to 1 positive"),
        ("up_to_3", "Up to 3 positives"),
        ("up_to_3_plus_3_negative", "Up to 3 positives\n+ 3 negatives"),
        ("full_supported", "All supported\nevidence"),
    ]
    expected_n = 53_386
    for key, _ in views:
        assert saved["test"][key]["V1"]["rows"] == expected_n
        assert saved["test"][key]["V3"]["rows"] == expected_n

    x = np.arange(len(views))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    colors = {"V1": "#a7a0ba", "V3": "#62428e"}
    for i, model in enumerate(("V1", "V3")):
        values = [100 * saved["test"][key][model]["top1_accuracy"] for key, _ in views]
        bars = ax.bar(x + (i - 0.5) * width, values, width, color=colors[model], label=model)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10, fontweight="bold")
    ax.set_xticks(x, [label for _, label in views])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 classification accuracy (%)")
    ax.set_title("Matched V1–V3 evaluation on partial evidence", loc="left", weight="bold", pad=20)
    ax.text(
        0,
        1.01,
        "Official synthetic test subset · 53,386 rows per view · same rows and evidence settings",
        transform=ax.transAxes,
        fontsize=10,
        color="#514a5d",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#ece8f2")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(OUT / "matched_v1_v3_accuracy.png", dpi=180)
    plt.close(fig)


def focused_candidate_confusion() -> None:
    with CONFUSION.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["view"] == "up_to_2"
            and row["cohort"] == "primary"
            and row["model"] == "rf_group_weighted"
        ]
    true_labels = ["Bronchiolitis", "Tuberculosis", "Pneumonia", "Influenza"]
    by_true = {label: [r for r in rows if r["actual_condition"] == label] for label in true_labels}
    assert all(by_true.values())
    for label, group in by_true.items():
        assert sum(int(r["count"]) for r in group) == int(group[0]["true_condition_total"]), label

    # Include every selected true class, the two largest off-diagonal destinations,
    # and one explicit remainder column so no prediction mass disappears.
    off_diagonal = Counter()
    for label in true_labels:
        for row in by_true[label]:
            predicted = row["predicted_condition"]
            if predicted not in true_labels:
                off_diagonal[predicted] += int(row["count"])
    columns = true_labels + [label for label, _ in off_diagonal.most_common(2)]
    assert len(columns) == len(set(columns)) == 6
    counts = []
    totals = []
    for label in true_labels:
        group = by_true[label]
        lookup = {r["predicted_condition"]: int(r["count"]) for r in group}
        total = int(group[0]["true_condition_total"])
        row_counts = [lookup.get(predicted, 0) for predicted in columns]
        row_counts.append(total - sum(row_counts))
        assert sum(row_counts) == total
        counts.append(row_counts)
        totals.append(total)
    percentages = np.array(counts) / np.array(totals)[:, None] * 100

    fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    im = ax.imshow(percentages, cmap="Purples", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(7), columns + ["Other\npredictions"])
    ax.set_yticks(range(4), [f"{label} (n={n:,})" for label, n in zip(true_labels, totals)])
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    ax.set_title("Selected rows of the candidate confusion matrix", loc="left", weight="bold", pad=22)
    ax.text(
        0,
        1.02,
        "Offline group-weighted candidate · up to 2 reporting units · primary test cohort",
        transform=ax.transAxes,
        fontsize=10,
        color="#514a5d",
    )
    for i in range(4):
        for j in range(7):
            color = "white" if percentages[i, j] >= 55 else "#241d31"
            ax.text(j, i, f"{percentages[i, j]:.1f}%\n({counts[i][j]:,})", ha="center", va="center", color=color, fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="Share of true-condition cases (%)", shrink=0.9)
    fig.savefig(OUT / "focused_candidate_confusion_up_to_2.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    matched_v1_v3()
    focused_candidate_confusion()
    print("Wrote two documentation figures from frozen evaluation records.")
