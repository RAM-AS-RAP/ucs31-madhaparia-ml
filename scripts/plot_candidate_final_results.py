"""Create documentation figures from verified final-evaluation results."""

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
REPORTS = ROOT / "reports/candidate_final_evaluation"
EVALUATION = REPORTS / "recorded_results/evaluation"
FIGURES = REPORTS / "figures"
TABLE = REPORTS / "final_evaluation_tables.md"
MANIFEST = REPORTS / "documentation_manifest.json"

VIEWS = ["full", "up_to_1", "up_to_2", "up_to_3", "up_to_5"]
LABELS = ["Full", "Up to 1", "Up to 2", "Up to 3", "Up to 5"]
MODELS = ["frozen_v2_harmonized", "rf_group_weighted"]
MODEL_LABELS = ["Frozen V2 (harmonized)", "Selected candidate"]
GRAY = "#687787"
BLUE = "#3274B5"
TEAL = "#007C83"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return "undefined" if pd.isna(value) else f"{100 * value:.2f}%"


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def main():
    if FIGURES.exists() or TABLE.exists() or MANIFEST.exists():
        raise FileExistsError("Final documentation output already exists.")

    export_path = REPORTS / "results_export_manifest.json"
    export = read_json(export_path)
    inputs = {export_path.resolve(): file_hash(export_path)}

    for relative, expected in export["output_sha256"].items():
        path = (REPORTS / relative).resolve()
        if not path.is_relative_to(REPORTS.resolve()):
            raise ValueError("Unexpected report path")
        if file_hash(path) != expected:
            raise ValueError(f"Exported result changed: {path}")
        inputs[path] = expected

    inputs[Path(__file__).resolve()] = file_hash(Path(__file__))

    evaluation = read_json(EVALUATION / "evaluation_manifest.json")
    groups = evaluation["cohort_counts"]["primary"]
    classes = sorted(evaluation["cohort_support"]["primary"])
    if not groups:
        raise ValueError("Primary cohort is not assessable")

    def primary_csv(filename):
        frame = pd.read_csv(EVALUATION / filename)
        selected = frame.loc[frame["cohort"].eq("primary")].copy()
        if set(selected["view"]) != set(VIEWS):
            raise ValueError(f"Unexpected primary-cohort views: {filename}")
        return selected

    classification = primary_csv("classification_summary.csv")
    policy = primary_csv("policy_summary.csv")
    probability = primary_csv("candidate_probability_summary.csv")
    reliability = primary_csv("reliability_bins.csv")
    per_condition = primary_csv("policy_per_condition.csv")
    class_conditions = primary_csv("classification_per_condition.csv")
    confusion = primary_csv("confusion_counts.csv")

    for frame in (classification, policy, probability):
        if not frame["observations"].eq(groups).all():
            raise ValueError("Summary denominator differs from primary cohort")

    if (
        len(classification) != 10
        or classification.duplicated(["view", "model"]).any()
        or set(classification["model"]) != set(MODELS)
    ):
        raise ValueError("Unexpected classification summary")

    if len(policy) != 5 or policy["view"].duplicated().any():
        raise ValueError("Unexpected policy summary")

    if (
        len(probability) != 10
        or probability.duplicated(["view", "representation"]).any()
        or set(probability["representation"]) != {"original", "temperature_scaled"}
    ):
        raise ValueError("Unexpected probability summary")

    classification = classification.set_index(["view", "model"])
    policy = policy.set_index("view").loc[VIEWS]
    probability = probability.set_index(["view", "representation"])

    original = probability.xs("original", level="representation").loc[VIEWS]
    scaled = probability.xs("temperature_scaled", level="representation").loc[VIEWS]
    np.testing.assert_allclose(
        original["accuracy"], scaled["accuracy"], rtol=0, atol=1e-12
    )

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "candidate-final-evaluation-issue-50",
    })

    FIGURES.mkdir(parents=True, exist_ok=False)
    outputs = {}

    def save(fig, name):
        for extension in ("png", "svg"):
            path = FIGURES / f"{name}.{extension}"
            options = (
                {"dpi": 180}
                if extension == "png"
                else {"metadata": {"Date": None}}
            )
            fig.savefig(path, bbox_inches="tight", **options)
            if extension == "svg":
                content = path.read_text(encoding="utf-8")
                path.write_text(
                    "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                    encoding="utf-8",
                )
            outputs[path.relative_to(REPORTS).as_posix()] = file_hash(path)
        plt.close(fig)

    x = np.arange(len(VIEWS))
    subtitle = f"Official test split; primary cohort: {groups:,} source groups"

    # 1. Unfiltered classification comparison.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, metric, title in zip(
        axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro F1"]
    ):
        for offset, model, label, color in zip(
            [-0.18, 0.18], MODELS, MODEL_LABELS, [GRAY, TEAL]
        ):
            values = [
                100 * classification.loc[(view, model), metric] for view in VIEWS
            ]
            bars = ax.bar(x + offset, values, 0.36, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_title(title)
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Percent; before withholding")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(subtitle)
    save(fig, "classification_comparison")

    # 2. Coverage and accepted accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    bars = axes[0].bar(x, 100 * policy["coverage"], color=TEAL)
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].set_title("Coverage: accepted / all eligible observations")

    accuracy = 100 * policy["accepted_accuracy"].to_numpy()
    lower = 100 * policy["accepted_accuracy_wilson_lower"].to_numpy()
    upper = 100 * policy["accepted_accuracy_wilson_upper"].to_numpy()
    available = np.isfinite(accuracy)

    axes[1].plot(
        x, 100 * policy["unfiltered_accuracy"], "o--",
        color=GRAY, label="Candidate accuracy on all observations",
    )
    axes[1].errorbar(
        x[available], accuracy[available],
        yerr=np.vstack([
            accuracy[available] - lower[available],
            upper[available] - accuracy[available],
        ]),
        fmt="o", capsize=4, color=TEAL,
        label="Accepted accuracy; 95% Wilson interval",
    )
    for index in np.flatnonzero(~available):
        axes[1].annotate(
            "No accepted cases\naccuracy undefined", (index, 8),
            ha="center", fontsize=9,
        )
    axes[1].set_title("Accuracy denominators differ")
    axes[1].legend(loc="lower right", fontsize=8)

    for ax in axes:
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 106)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(subtitle)
    save(fig, "coverage_and_accuracy")

    # 3. Calibration metrics.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["log_loss", "multiclass_brier_score", "top_label_ece"],
        ["Log loss", "Multiclass Brier score", "Top-label ECE (10 bins)"],
    ):
        ax.bar(x - 0.18, original[metric], 0.36, color=GRAY, label="Original")
        ax.bar(
            x + 0.18, scaled[metric], 0.36,
            color=TEAL, label="Temperature scaled",
        )
        ax.set_title(title)
        ax.set_xticks(x, LABELS, rotation=25)
        ax.set_ylabel("Metric value; lower is better")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle(subtitle)
    save(fig, "calibration_metrics")

    # 4. Reliability diagrams.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    for ax, view, label in zip(axes.flat, VIEWS, LABELS):
        ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", label="Identity")
        for representation, name, color in (
            ("original", "Original", GRAY),
            ("temperature_scaled", "Temperature scaled", TEAL),
        ):
            rows = reliability[
                reliability["view"].eq(view)
                & reliability["representation"].eq(representation)
                & reliability["count"].gt(0)
            ].sort_values("bin_index")
            ax.plot(
                rows["mean_top_score"], rows["accuracy"], "o-",
                color=color, label=name,
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
        "Ten equal-width bins; empty bins omitted.\n\n"
        "Bin counts are retained in reliability_bins.csv.\n"
        "Lines connect summaries, not fitted curves.\n\n"
        "Views reuse the same source groups.\n"
        "These are simulated-data results.",
        va="top", linespacing=1.7,
    )
    fig.suptitle(subtitle)
    save(fig, "reliability_diagrams")

    # 5. Per-condition recall, including withheld true cases.
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 13), sharey=True, layout="constrained"
    )
    y = np.arange(len(classes))

    for ax, view, label in zip(
        axes, ["full", "up_to_2"], ["Full view", "Up to 2 reporting units"]
    ):
        rows = per_condition[
            per_condition["view"].eq(view)
        ].set_index("condition").loc[classes]
        baseline = class_conditions[
            class_conditions["view"].eq(view)
            & class_conditions["model"].eq("frozen_v2_harmonized")
        ].set_index("condition").loc[classes]

        ax.barh(
            y - 0.24, 100 * baseline["recall"], 0.24,
            color=GRAY, label="V2 recall before withholding",
        )
        ax.barh(
            y, 100 * rows["unfiltered_recall"], 0.24,
            color=BLUE, label="Candidate recall before withholding",
        )
        ax.barh(
            y + 0.24, 100 * rows["correct_accepted_recall_all_cases"], 0.24,
            color=TEAL, label="Candidate correct accepted / all true cases",
        )
        ax.set_title(label)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Percent of all true cases of each condition")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    support = evaluation["cohort_support"]["primary"]
    axes[0].set_yticks(y, [
        textwrap.fill(f"{name} (n={support[name]})", 36) for name in classes
    ])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(subtitle)
    save(fig, "per_condition_recall")

    # 6–7. Unfiltered confusion matrices, retaining V2's COVID-19 output.
    for view, title in (("full", "Full view"), ("up_to_2", "Up to 2 units")):
        fig, axes = plt.subplots(
            1, 2, figsize=(25, 12), layout="constrained"
        )
        for ax, model, model_label in zip(axes, MODELS, MODEL_LABELS):
            rows = confusion[
                confusion["view"].eq(view) & confusion["model"].eq(model)
            ]
            predictions = sorted(
                set(classes) | ({"COVID-19"} if model == MODELS[0] else set())
            )
            counts = rows.pivot(
                index="actual_condition",
                columns="predicted_condition",
                values="count",
            ).reindex(index=classes, columns=predictions)

            if counts.isna().any().any():
                raise ValueError("Missing confusion-matrix entries")
            if int(counts.to_numpy().sum()) != groups:
                raise ValueError("Confusion matrix lost primary observations")

            values = counts.to_numpy(dtype=float)
            totals = values.sum(axis=1, keepdims=True)
            normalized = np.divide(
                100 * values, totals,
                out=np.full_like(values, np.nan), where=totals > 0,
            )

            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#EEEEEE")
            picture = ax.imshow(
                np.ma.masked_invalid(normalized),
                vmin=0, vmax=100, cmap=cmap, aspect="auto",
            )
            ax.set_xticks(
                np.arange(len(predictions)),
                [textwrap.fill(name, 22) for name in predictions],
                rotation=60, ha="right", fontsize=8,
            )
            ax.set_yticks(
                np.arange(len(classes)),
                [textwrap.fill(name, 25) for name in classes],
                fontsize=8,
            )
            for row_index in range(len(classes)):
                for column_index in range(len(predictions)):
                    value = normalized[row_index, column_index]
                    if np.isfinite(value) and value >= 0.5:
                        ax.text(
                            column_index, row_index, f"{value:.1f}",
                            ha="center", va="center", fontsize=6,
                            color="white" if value > 50 else "black",
                        )

            ax.set_title(model_label)
            ax.set_xlabel("Predicted condition")
            ax.set_ylabel("Actual condition")
            fig.colorbar(
                picture, ax=ax, shrink=0.65,
                label="Percent within actual condition",
            )

        fig.suptitle(
            f"{title}: unfiltered classification\n{subtitle}\n"
            "V2 retains a COVID-19 prediction column; no COVID-19 true cases are present."
        )
        save(fig, f"confusion_{view}")

    # Tables for direct use in documentation.
    sections = [
        "# Final evaluation tables",
        f"Primary cohort: {groups:,} eligible unique source-input groups from "
        "the official test split. Earlier project-level test exposure is "
        "documented in CANDIDATE_FINAL_EVALUATION_PROTOCOL.md. "
        "This is not claimed to be a never-inspected project-wide test.",
        "## Classification before withholding",
        "Precision, recall and F1 are macro averages over the fixed "
        "21 ground-truth conditions.",
        markdown_table(
            ["View", "Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1"],
            [
                [
                    label, model_label,
                    *[
                        percent(classification.loc[(view, model), metric])
                        for metric in [
                            "accuracy", "macro_precision", "macro_recall", "macro_f1"
                        ]
                    ],
                ]
                for view, label in zip(VIEWS, LABELS)
                for model, model_label in zip(MODELS, MODEL_LABELS)
            ],
        ),
        "## Candidate policy",
        "Accepted accuracy excludes withheld cases. Coverage uses all eligible "
        "cases. Intervals are two-sided 95% Wilson accuracy intervals.",
        markdown_table(
            ["View", "Accepted / eligible", "Coverage", "Accepted accuracy",
             "Accuracy interval", "Correct accepted / all"],
            [
                [
                    label,
                    f"{int(policy.loc[view, 'accepted']):,} / {groups:,}",
                    percent(policy.loc[view, "coverage"]),
                    percent(policy.loc[view, "accepted_accuracy"]),
                    (
                        "undefined"
                        if pd.isna(policy.loc[view, "accepted_accuracy"])
                        else
                        f"{percent(policy.loc[view, 'accepted_accuracy_wilson_lower'])}"
                        f" to {percent(policy.loc[view, 'accepted_accuracy_wilson_upper'])}"
                    ),
                    percent(policy.loc[view, "correct_accepted_fraction_of_all"]),
                ]
                for view, label in zip(VIEWS, LABELS)
            ],
        ),
        "## Candidate calibration",
        "Lower is better. Brier score is summed across classes. "
        "ECE uses ten equal-width top-label score bins.",
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
                for representation in ("original", "temperature_scaled")
            ],
        ),
        "## Interpretation",
        "- Complete eligible and overlap-cohort results remain in the recorded CSVs.\n"
        "- The separately recorded zero-supported group is outside these denominators.\n"
        "- Repeated views reuse the same groups and are not independent patients.\n"
        "- Aggregate target success does not establish adequate recall for each condition.\n"
        "- Per-condition correct-accepted recall includes withheld true cases.\n"
        "- One-unit ECE worsened after temperature scaling.\n"
        "- Sparse tuberculosis and bronchiolitis results and complete Larygospasm "
        "withholding remain limitations.\n"
        "- The earlier unlabelled chest-pain probe remains a documented concern.\n"
        "- These simulated-data results do not establish clinical performance "
        "for Kenyan users or authorize deployment.",
    ]
    TABLE.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    outputs[TABLE.relative_to(REPORTS).as_posix()] = file_hash(TABLE)

    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Documentation input changed: {path}")

    manifest = {
        "issue": 50,
        "status": "final_documentation_figures_and_tables_exported",
        "cohort": "primary",
        "source_groups": groups,
        "input_sha256": {
            path.relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in inputs.items()
        },
        "output_sha256": outputs,
        "matplotlib_version": matplotlib.__version__,
        "notes": [
            "Uses exported verified results; no predictions recalculated.",
            "Classification and confusion plots precede withholding.",
            "Coverage and accepted accuracy retain different denominators.",
            "Undefined accepted accuracy is not plotted as zero.",
            "Confusion values are percentages within actual-condition rows.",
            "Confusion annotations below 0.5 percent are omitted; CSV counts remain complete.",
            "V2's COVID-19 prediction column is retained.",
            "SVG trailing whitespace was removed before fingerprinting.",
        ],
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL DOCUMENTATION EXPORT COMPLETE")
    print("Primary source groups:", groups)
    print("Figures: 7 PNG and 7 SVG")
    print("Table:", TABLE.name)
    print("Input fingerprints unchanged.")
    print("No predictions calculated.")
    print("Saved to:", REPORTS)


if __name__ == "__main__":
    main()