# Evaluation

This folder contains Jupyter notebooks for explaining the project evaluation.

The teacher specifically mentioned F1-score, so the notebooks focus on:

- accuracy
- precision
- recall
- macro F1
- weighted F1
- per-label F1
- model comparison charts

## Notebooks

- `evaluate.ipynb`: explains F1-score and shows the final evaluation result.
- `compare_models.ipynb`: compares Traditional ML and Deep Learning / DeBERTa with charts.
- `fresh_model_comparison.ipynb`: reruns evaluation on the current dataset using
  the current Rule-based, Traditional ML, and Deep Learning / DeBERTa methods.

Both notebooks read the saved report files:

- `backends/traditional_ml/training_report.json`
- `backends/deep_learning/bert_training_report.json`

The fresh comparison notebook is different: it loads the current saved models
and recomputes predictions on the processed dataset, so it can produce new
charts without retraining.

Open the notebooks from the project root or from this folder, then run all cells.

The main comparison metric is **macro F1** because the dataset has many `Unknown`
labels and is imbalanced.
