"""Create documentation figures from the exported calibration results."""

import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/candidate_calibration"
RECORDED = REPORTS / "recorded_results"
EVALUATION = RECORDED / "uncertainty_evaluation"
FIGURES = REPORTS / "figures"
TABLE = REPORTS / "evaluation_tables.md"
MANIFEST = REPORTS / "documentation_manifest.json"

VIEWS = ["full", "up_to_1", "up_to_2", "up_to_3", "up_to_5"]
LABELS = ["Full", "Up to 1", "Up to 2", "Up to 3", "Up to 5"]
COLORS = {"original": "#687787", "temperature_scaled": "#007C83"}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return "undefined" if pd.isna(value) else f"{100 * value:.2f}%"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main():
    if FIGURES.exists() or TABLE.exists() or MANIFEST.exists():
        raise FileExistsError("Documentation output already exists.")

    export_path = REPORTS / "results_export_manifest.json"
    export = read_json(export_path)
    inputs = {export_path: file_hash(export_path)}

    for relative, expected in export["output_sha256"].items():
        path = (REPORTS / relative).resolve()
        if not path.is_relative_to(REPORTS.resolve()):
            raise ValueError("Export path is outside the report folder")
        if file_hash(path) != expected:
            raise RuntimeError(f"Exported result changed: {path}")
        inputs[path] = expected

    inputs[Path(__file__).resolve()] = file_hash(Path(__file__))

    probability = pd.read_csv(EVALUATION / "probability_summary.csv")
    selective = pd.read_csv(EVALUATION / "policy_summary.csv")
    reliability = pd.read_csv(EVALUATION / "reliability_bins.csv")
    conditions = pd.read_csv(EVALUATION / "per_condition.csv")
    policy = read_json(RECORDED / "uncertainty_policy/policy.json")

    groups = export["evaluation_groups_per_view"]

    for frame in (probability, selective, reliability, conditions):
        if set(frame["partition"]) != {"uncertainty_evaluation"}:
            raise ValueError("Unexpected evaluation partition")
        if set(frame["view"]) != set(VIEWS):
            raise ValueError("Unexpected evaluation views")

    if probability.duplicated(["view", "representation"]).any():
        raise ValueError("Duplicate probability summaries")

    representations = {"original", "temperature_scaled"}
    if len(probability) != 10 or set(probability["representation"]) != representations:
        raise ValueError("Expected both probability representations in all views")

    if len(selective) != 5 or selective["view"].duplicated().any():
        raise ValueError("Expected one policy summary per view")

    if not (probability["observations"] == groups).all():
        raise ValueError("Probability observation counts differ")
    if not (selective["observations"] == groups).all():
        raise ValueError("Policy observation counts differ")

    if conditions.duplicated(["view", "condition"]).any():
        raise ValueError("Duplicate per-condition rows")
    for view in VIEWS:
        rows = conditions[conditions["view"] == view]
        if set(rows["condition"]) != set(policy["classes"]):
            raise ValueError(f"Condition coverage differs: {view}")

    probability = probability.set_index(["view", "representation"])
    selective = selective.set_index("view").loc[VIEWS]

    original = probability.xs("original", level="representation").loc[VIEWS]
    scaled = probability.xs(
        "temperature_scaled", level="representation"
    ).loc[VIEWS]

    if not np.allclose(
        original["accuracy"], scaled["accuracy"], rtol=0, atol=1e-12
    ):
        raise ValueError("Temperature scaling changed recorded accuracy")

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "candidate-calibration-issue-49",
    })

    FIGURES.mkdir(parents=True, exist_ok=False)
    outputs = {}

    def save_figure(fig, name):
        for extension in ("png", "svg"):
            path = FIGURES / f"{name}.{extension}"
            options = {"dpi": 180} if extension == "png" else {
                "metadata": {"Date": None}
            }
            fig.savefig(path, bbox_inches="tight", **options)

            if extension == "svg":
                # Remove generated trailing whitespace before fingerprinting.
                content = path.read_text(encoding="utf-8")
                path.write_text(
                    "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                    encoding="utf-8",
                )

            outputs[path.relative_to(REPORTS).as_posix()] = file_hash(path)
        plt.close(fig)

    x = np.arange(len(VIEWS))

    # 1. Calibration metrics: lower values are better.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["log_loss", "multiclass_brier_score", "top_label_ece"],
        ["Log loss", "Multiclass Brier score", "Top-label ECE (10 bins)"],
    ):
        ax.bar(x - 0.18, original[metric], 0.36,
               label="Original", color=COLORS["original"])
        ax.bar(x + 0.18, scaled[metric], 0.36,
               label="Temperature scaled", color=COLORS["temperature_scaled"])
        ax.set_xticks(x, LABELS, rotation=25)
        ax.set_title(title)
        ax.set_ylabel("Metric value; lower is better")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle(
        f"Frozen calibration evaluation: {groups:,} simulated source groups per view"
    )
    save_figure(fig, "calibration_metrics")

    # 2. Coverage and accuracy, showing different denominators explicitly.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

    coverage = 100 * selective["coverage"].to_numpy()
    bars = axes[0].bar(x, coverage, color="#007C83")
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].set_title("Coverage: accepted / all observations")

    accuracy = 100 * selective["accepted_accuracy"].to_numpy()
    lower = 100 * selective["accepted_accuracy_wilson_lower"].to_numpy()
    upper = 100 * selective["accepted_accuracy_wilson_upper"].to_numpy()
    available = np.isfinite(accuracy)

    axes[1].plot(
        x, 100 * selective["unfiltered_accuracy"],
        "o--", color="#687787", label="Accuracy on all observations",
    )
    axes[1].errorbar(
        x[available], accuracy[available],
        yerr=np.vstack([
            accuracy[available] - lower[available],
            upper[available] - accuracy[available],
        ]),
        fmt="o", capsize=4, color="#007C83",
        label="Accepted accuracy; 95% Wilson interval",
    )
    for index in np.flatnonzero(~available):
        axes[1].annotate(
            "No accepted cases\naccuracy undefined",
            (index, 8), ha="center", fontsize=9,
        )
    axes[1].set_title("Accuracy: all cases versus accepted cases")
    axes[1].legend(loc="lower right", fontsize=8)

    for ax in axes:
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 106)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle("Frozen uncertainty policy on the reserved evaluation partition")
    save_figure(fig, "coverage_and_accuracy")

    # 3. Reliability diagrams, excluding empty bins.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    for ax, view, label in zip(axes.flat, VIEWS, LABELS):
        ax.plot([0, 1], [0, 1], "--", color="#A0A0A0", label="Identity")
        for representation, name in (
            ("original", "Original"),
            ("temperature_scaled", "Temperature scaled"),
        ):
            rows = reliability[
                (reliability["view"] == view)
                & (reliability["representation"] == representation)
                & (reliability["count"] > 0)
            ].sort_values("bin_index")
            ax.plot(
                rows["mean_top_score"], rows["accuracy"], "o-",
                color=COLORS[representation], label=name,
            )
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean leading score in bin")
        ax.set_ylabel("Observed accuracy in bin")
        ax.grid(alpha=0.2)

    axes.flat[0].legend(fontsize=8)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0, 0.9,
        "Ten equal-width score bins.\nEmpty bins are omitted.\n\n"
        "Bin counts are recorded in reliability_bins.csv.\n"
        "Lines connect bin summaries; they are not fitted curves.\n\n"
        "All views reuse the same evaluation groups.\n"
        "These are simulated-data results.",
        va="top", linespacing=1.7,
    )
    fig.suptitle("Top-label reliability on the reserved evaluation partition")
    save_figure(fig, "reliability_diagrams")

    # 4. Per-condition recall keeps withheld true cases in the denominator.
    names = sorted(policy["classes"])
    fig, axes = plt.subplots(
        1, 2, figsize=(17, 12), sharey=True, layout="constrained"
    )
    y = np.arange(len(names))

    for ax, view, title in zip(
        axes, ["full", "up_to_2"], ["Full view", "Up to 2 reporting units"]
    ):
        rows = conditions[conditions["view"] == view].set_index("condition").loc[names]
        ax.barh(
            y - 0.18, 100 * rows["unfiltered_recall"], height=0.36,
            color="#687787", label="Recall before withholding",
        )
        ax.barh(
            y + 0.18, 100 * rows["correct_accepted_recall_all_cases"],
            height=0.36, color="#007C83",
            label="Correct accepted / all true cases",
        )
        ax.set_title(title)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Percent of all true cases of each condition")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    support = conditions[conditions["view"] == "full"].set_index("condition")
    axes[0].set_yticks(y, [
        textwrap.fill(f"{name} (n={int(support.loc[name, 'support_all'])})", 36)
        for name in names
    ])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-condition recall, including withheld cases in the denominator")
    save_figure(fig, "per_condition_recall")

    # Documentation tables use recorded values without recomputing predictions.
    sections = [
        "# Calibration and uncertainty evaluation",
        f"{groups:,} reserved simulated source groups per view, across 21 conditions. "
        "The five views reuse the same groups and are not independent samples. "
        "These results are separate from the earlier model-selection validation "
        "results and from the final test.",
        "## Classification before withholding",
        "Temperature scaling preserves the leading class. Precision, recall and "
        "F1 below are macro averages across the 21 conditions.",
        markdown_table(
            ["View", "Accuracy", "Macro precision", "Macro recall", "Macro F1"],
            [
                [label, *[
                    percent(scaled.loc[view, metric])
                    for metric in [
                        "accuracy", "macro_precision", "macro_recall", "macro_f1"
                    ]
                ]]
                for view, label in zip(VIEWS, LABELS)
            ],
        ),
        "## Frozen policy results",
        "Accepted accuracy uses only accepted cases. Coverage uses all cases. "
        "The interval is a two-sided 95% Wilson interval.",
        markdown_table(
            ["View", "Accepted / total", "Coverage", "Accepted accuracy",
             "Accuracy interval", "Correct accepted / all"],
            [
                [
                    label,
                    f"{int(selective.loc[view, 'accepted']):,} / {groups:,}",
                    percent(selective.loc[view, "coverage"]),
                    percent(selective.loc[view, "accepted_accuracy"]),
                    (
                        "undefined"
                        if pd.isna(selective.loc[view, "accepted_accuracy"])
                        else
                        f"{percent(selective.loc[view, 'accepted_accuracy_wilson_lower'])}"
                        f" to {percent(selective.loc[view, 'accepted_accuracy_wilson_upper'])}"
                    ),
                    percent(selective.loc[view, "correct_accepted_fraction_of_all"]),
                ]
                for view, label in zip(VIEWS, LABELS)
            ],
        ),
        "## Calibration comparison",
        "Lower values are better. Brier score is summed across classes. "
        "ECE is top-label ECE with ten equal-width bins.",
        markdown_table(
            ["View", "Representation", "Log loss", "Brier score", "ECE"],
            [
                [
                    label, representation,
                    *[
                        f"{probability.loc[(view, representation), metric]:.6f}"
                        for metric in [
                            "log_loss", "multiclass_brier_score", "top_label_ece"
                        ]
                    ],
                ]
                for view, label in zip(VIEWS, LABELS)
                for representation in ["original", "temperature_scaled"]
            ],
        ),
        "## Interpretation limits",
        "- Aggregate targets do not guarantee performance for each condition.\n"
        "- Accepted-case accuracy must be reported alongside coverage and "
        "all-case per-condition recall.\n"
        "- Withheld cases remain in the denominator of correct-accepted recall.\n"
        "- Temperature scaling does not improve every calibration metric in "
        "every view; inspect the one-unit ECE results.\n"
        "- The unlabelled tiredness/chest-pain probe still returns pulmonary "
        "neoplasm at approximately 78.02%, just above the score threshold. "
        "This is a remaining limitation, not evidence of diagnostic correctness.\n"
        "- Probe handling does not constitute a complete application safety "
        "assessment.\n"
        "- Clinical performance for Kenyan users has not been established.",
    ]

    TABLE.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    outputs[TABLE.relative_to(REPORTS).as_posix()] = file_hash(TABLE)

    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Documentation input changed: {path}")

    manifest = {
        "issue": 49,
        "status": "documentation_figures_and_tables_exported",
        "partition": "uncertainty_evaluation",
        "groups_per_view": groups,
        "input_sha256": {
            path.resolve().relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in inputs.items()
        },
        "output_sha256": outputs,
        "matplotlib_version": matplotlib.__version__,
        "notes": [
            "Figures and tables use previously exported metrics.",
            "No predictions were calculated and no policy was changed.",
            "Undefined accepted accuracy is not plotted as zero.",
            "SVG trailing whitespace was removed before fingerprinting.",
        ],
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nCALIBRATION DOCUMENTATION COMPLETE")
    print("Evaluation groups per view:", groups)
    print("Figures: 4 PNG and 4 SVG")
    print("Table:", TABLE.name)
    print("Input fingerprints unchanged.")
    print("Saved to:", REPORTS)


if __name__ == "__main__":
    main()