"""Export comparison documentation and figures from verified saved results."""

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
SOURCE = EXPERIMENT / "comparison"
OUTPUT = ROOT / "reports/candidate_model_comparison"

MODELS = [
    "frozen_v2_harmonized",
    "rf_group_weighted",
    "rf_class_balanced",
]
DISPLAY_NAMES = {
    "frozen_v2_harmonized": "Frozen V2 (harmonized)",
    "rf_group_weighted": "Candidate: group weighted",
    "rf_class_balanced": "Candidate: class balanced",
}
COLORS = ["#64748b", "#0072B2", "#E69F00"]
SETTINGS = ["up_to_1", "up_to_2", "up_to_3", "up_to_5", "full"]
VIEW_LABELS = ["Up to 1", "Up to 2", "Up to 3", "Up to 5", "Full"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, stem):
    for extension in ("png", "svg"):
        fig.savefig(
            OUTPUT / "figures" / f"{stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Documentation directory exists: {OUTPUT}")

    comparison = read_json(SOURCE / "comparison_manifest.json")
    selection = read_json(SOURCE / "selection.json")

    for filename, expected in comparison["report_sha256"].items():
        if file_hash(SOURCE / filename) != expected:
            raise ValueError(f"Comparison report changed: {filename}")

    summary = pd.read_csv(SOURCE / "summary.csv")
    per_condition = pd.read_csv(SOURCE / "per_condition.csv")
    confusion = pd.read_csv(SOURCE / "confusion_counts.csv")

    primary = summary.loc[summary["cohort"].eq("selection_primary")]
    if primary["observations"].nunique() != 1:
        raise ValueError("Primary cohort sizes differ")

    group_count = int(primary.iloc[0]["observations"])
    selected = selection["selected_candidate"]

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "figures").mkdir()

    for filename in [
        *comparison["report_sha256"],
        "comparison_manifest.json",
    ]:
        shutil.copyfile(SOURCE / filename, OUTPUT / filename)

    additional_reports = {
        "partition_manifest.json": (
            EXPERIMENT / "partition_manifest.json",
            "partition_manifest",
        ),
        "view_manifest.json": (
            EXPERIMENT / "validation_views/view_manifest.json",
            "view_manifest",
        ),
        "rf_group_weighted_training_manifest.json": (
            EXPERIMENT / "models/rf_group_weighted/training_manifest.json",
            "rf_group_weighted_manifest",
        ),
        "rf_class_balanced_training_manifest.json": (
            EXPERIMENT / "models/rf_class_balanced/training_manifest.json",
            "rf_class_balanced_manifest",
        ),
    }

    for filename, (path, fingerprint_key) in additional_reports.items():
        if file_hash(path) != comparison["input_sha256"][fingerprint_key]:
            raise ValueError(f"Supporting manifest changed: {filename}")
        shutil.copyfile(path, OUTPUT / filename)

    # Aggregate charts include recall so the accuracy/F1 view is not selective.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    x = np.arange(len(SETTINGS))
    width = 0.24

    for ax, metric, title in zip(
        axes,
        ["accuracy", "macro_f1", "macro_recall"],
        ["Accuracy", "Macro F1", "Macro recall"],
    ):
        for index, model in enumerate(MODELS):
            rows = primary.loc[primary["model"].eq(model)].set_index("view")
            values = rows.loc[SETTINGS, metric].to_numpy() * 100

            ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=DISPLAY_NAMES[model],
                color=COLORS[index],
            )

        ax.set_title(title)
        ax.set_xticks(x, VIEW_LABELS)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percent")
        ax.set_xlabel("Available reporting units")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False,
    )
    fig.suptitle(
        f"Primary validation comparison: {group_count:,} source-input groups",
        y=1.0,
    )
    fig.subplots_adjust(top=0.79, bottom=0.19, wspace=0.28)
    fig.text(
        0.5, 0.02,
        "Simulated DDXPlus data; shared observation views; "
        "these are validation results, not clinical accuracy.",
        ha="center", fontsize=9,
    )
    save_figure(fig, "primary_metrics")

    # Per-condition changes, including regressions.
    condition_primary = per_condition.loc[
        per_condition["cohort"].eq("selection_primary")
        & per_condition["support"].gt(0)
    ]

    baseline = condition_primary.loc[
        condition_primary["model"].eq("frozen_v2_harmonized")
    ]
    candidate = condition_primary.loc[
        condition_primary["model"].eq(selected)
    ]

    deltas = baseline.merge(
        candidate,
        on=["view", "condition"],
        suffixes=("_v2", "_candidate"),
        validate="one_to_one",
    )

    if not deltas["support_v2"].equals(deltas["support_candidate"]):
        raise ValueError("Per-condition supports differ")

    for metric in ("precision", "recall", "f1"):
        deltas[f"{metric}_change_pp"] = 100 * (
            deltas[f"{metric}_candidate"] - deltas[f"{metric}_v2"]
        )

    delta_columns = [
        "view", "condition", "support_v2",
        "precision_v2", "precision_candidate", "precision_change_pp",
        "recall_v2", "recall_candidate", "recall_change_pp",
        "f1_v2", "f1_candidate", "f1_change_pp",
    ]
    deltas[delta_columns].to_csv(
        OUTPUT / "selected_vs_v2_per_condition.csv",
        index=False, lineterminator="\n",
    )

    support = (
        candidate.loc[candidate["view"].eq("full")]
        .set_index("condition")["support"]
    )

    for metric, title in (("f1", "F1"), ("recall", "Recall")):
        table = deltas.pivot(
            index="condition", columns="view",
            values=f"{metric}_change_pp",
        ).reindex(columns=SETTINGS).sort_index()

        values = table.to_numpy()
        bound = max(1.0, float(np.nanmax(np.abs(values))))

        fig, ax = plt.subplots(figsize=(12, 11))
        image = ax.imshow(
            values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto"
        )
        ax.set_xticks(np.arange(len(SETTINGS)), VIEW_LABELS)
        ax.set_yticks(
            np.arange(len(table)),
            [f"{name} (n={int(support[name])})" for name in table.index],
            fontsize=8,
        )

        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column, row, f"{value:+.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 * bound else "black",
                )

        ax.set_title(
            f"{title} change: selected candidate minus harmonized V2\n"
            "Primary validation cohort; positive values indicate improvement"
        )
        fig.colorbar(image, ax=ax, label="Percentage-point change")
        fig.tight_layout()
        save_figure(fig, f"per_condition_{metric}_change")

    # Confusion charts: row-normalized percentages with source-group support.
    truth_labels = comparison["ground_truth_labels"]
    output_labels = comparison["harmonized_output_labels"]

    for model in ("frozen_v2_harmonized", selected):
        for setting in ("full", "up_to_2"):
            rows = confusion.loc[
                confusion["cohort"].eq("selection_primary")
                & confusion["model"].eq(model)
                & confusion["view"].eq(setting)
            ]

            matrix = rows.pivot(
                index="actual_condition",
                columns="predicted_condition",
                values="count",
            ).reindex(index=truth_labels, columns=output_labels).to_numpy()

            totals = matrix.sum(axis=1)
            normalized = np.divide(
                100 * matrix,
                totals[:, None],
                out=np.zeros_like(matrix, dtype=float),
                where=totals[:, None] != 0,
            )

            fig, ax = plt.subplots(figsize=(14, 12))
            image = ax.imshow(
                normalized, vmin=0, vmax=100, cmap="Blues", aspect="auto"
            )
            ax.set_xticks(
                np.arange(len(output_labels)),
                output_labels,
                rotation=70, ha="right", fontsize=7,
            )
            ax.set_yticks(
                np.arange(len(truth_labels)),
                [
                    f"{label} (n={int(count)})"
                    for label, count in zip(truth_labels, totals)
                ],
                fontsize=8,
            )
            ax.set_xlabel("Predicted condition")
            ax.set_ylabel("Actual condition")
            ax.set_title(
                f"{DISPLAY_NAMES[model]} — {setting}\n"
                "Primary validation cohort; percentage of each actual class"
            )
            fig.colorbar(image, ax=ax, label="Row percentage")
            fig.tight_layout()
            save_figure(fig, f"confusion_{model}_{setting}")

    files = sorted(
        path for path in OUTPUT.rglob("*") if path.is_file()
    )
    documentation_manifest = {
        "comparison_manifest_sha256": file_hash(
            SOURCE / "comparison_manifest.json"
        ),
        "plot_script_sha256": file_hash(Path(__file__).resolve()),
        "matplotlib_version": matplotlib.__version__,
        "selected_candidate": selected,
        "primary_source_groups": group_count,
        "output_sha256": {
            path.relative_to(OUTPUT).as_posix(): file_hash(path)
            for path in files
        },
        "notes": [
            "Charts were generated from saved comparison metrics.",
            "No models were trained or predictions recomputed.",
            "Negative per-condition changes are retained in the charts.",
            "PNG files are exported at 300 dpi; SVG files are scalable.",
        ],
    }

    (OUTPUT / "documentation_manifest.json").write_text(
        json.dumps(documentation_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("DOCUMENTATION EXPORT COMPLETE")
    print("Selected candidate:", selected)
    print("Primary validation groups:", group_count)
    print("Charts: aggregate metrics, F1/recall changes, and confusion matrices.")
    print("Formats: PNG and SVG")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()