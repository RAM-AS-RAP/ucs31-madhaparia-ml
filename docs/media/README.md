# Documentation figure provenance

The two PNGs here are **presentation-only replots** of already saved, committed evaluation results. Run `python docs/media/build_ml_figures.py` to regenerate them. The script reads JSON/CSV and never loads a model or source dataset.

| Figure | Exact input | What is shown |
| --- | --- | --- |
| `matched_v1_v3_accuracy.png` | [`reports/release_model_evaluation/evaluation.json`](../../reports/release_model_evaluation/evaluation.json), `test` → each view → `V1`/`V3` → `top1_accuracy` | Four paired top-1 bars; both models use the same 53,386 eligible rows in each view. Values are rounded to one decimal **only in chart labels**. |
| `focused_candidate_confusion_up_to_2.png` | [`reports/candidate_final_evaluation/recorded_results/evaluation/confusion_counts.csv`](../../reports/candidate_final_evaluation/recorded_results/evaluation/confusion_counts.csv), `view=up_to_2`, `cohort=primary`, `model=rf_group_weighted` | Four selected true-condition rows. Columns include those true labels, the two most frequent other predicted labels across the selected rows, and an explicit remainder. Each row retains its full `true_condition_total`. Cells show raw counts and row percentages rounded to one decimal. |

The full original confusion matrix and all other recorded results remain unchanged. These plots are **not new model evaluations**. All source records are simulated; neither plot is a clinical validation or patient-probability estimate.
