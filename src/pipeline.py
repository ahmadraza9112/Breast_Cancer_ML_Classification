"""Leakage-aware classical ML benchmark on Wisconsin Diagnostic Breast Cancer data.

The target is explicitly mapped to 1=malignant and 0=benign. Features are
precomputed numerical descriptors, NOT raw medical images. This demonstration
is not for clinical use.
"""
from __future__ import annotations

import json
import platform
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
POSITIVE_CLASS = "malignant"
CV_FOLDS = 5
TEST_SIZE = 0.20


def get_data():
    """Return X, y and raw dataset metadata. y=1 denotes malignant."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = pd.Series((dataset.target == 0).astype(int), name="malignant", index=X.index)
    assert X.shape == (569, 30), "Unexpected dataset version"
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 212
    return X, y, dataset


def split_data(X, y):
    """Reproducible stratified split; retain original row indices for audit."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    return tr, te


def make_models():
    """Predetermined four comparison models. Scaling/PCA always inside Pipeline."""
    def logistic(pca=False):
        steps = [("scale", StandardScaler())]
        if pca:
            steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
        steps.append(("model", LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)))
        return Pipeline(steps)

    def svm(pca=False):
        steps = [("scale", StandardScaler())]
        if pca:
            steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
        steps.append(("model", SVC(C=1.0, kernel="rbf", gamma="scale")))
        return Pipeline(steps)

    return OrderedDict([
        ("Logistic Regression", logistic()),
        ("RBF SVM", svm()),
        ("Logistic Regression + PCA", logistic(pca=True)),
        ("RBF SVM + PCA", svm(pca=True)),
    ])


def cross_validate_models(models, X_train, y_train):
    """Select models on training data only. PCA learns within each CV fold."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = {
        "auc": "roc_auc", "balanced_accuracy": "balanced_accuracy",
        "f1_malignant": "f1", "accuracy": "accuracy",
    }
    rows = []
    for name, pipeline in models.items():
        result = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring=scores, n_jobs=1, error_score="raise")
        row = {"model": name}
        for key in scores:
            row[f"cv_{key}_mean"] = float(np.mean(result[f"test_{key}"]))
            row[f"cv_{key}_std"] = float(np.std(result[f"test_{key}"], ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(cv_table):
    """Choose highest CV ROC-AUC; tie-break CV balanced accuracy, then name."""
    ranked = cv_table.sort_values(
        by=["cv_auc_mean", "cv_balanced_accuracy_mean", "model"],
        ascending=[False, False, True], kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])


def eval_selected(estimator, X_test, y_test):
    """Evaluate exactly one selected estimator on previously untouched test data."""
    pred = estimator.predict(X_test)
    score = estimator.decision_function(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    m = {
        "test_n": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "precision_malignant": float(precision_score(y_test, pred, zero_division=0)),
        "recall_malignant_sensitivity": float(recall_score(y_test, pred, zero_division=0)),
        "f1_malignant": float(f1_score(y_test, pred, zero_division=0)),
        "specificity_benign": float(tn / (tn + fp)),
        "true_benign": int(tn),
        "false_malignant": int(fp),
        "missed_malignant": int(fn),
        "correct_malignant": int(tp),
    }
    return m, np.asarray(pred), np.asarray(score)


def plot_dataset(y, out):
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars = ax.bar(["Benign (0)", "Malignant (1)"], counts, color=["#7199c0", "#d77e72"])
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(counts)*1.18)
    ax.set_ylabel("Number of observations")
    ax.set_title("Dataset class distribution")
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def plot_cv(cv_table, out):
    d = cv_table.sort_values("cv_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(d))
    ax.errorbar(d["cv_auc_mean"], y_pos, xerr=d["cv_auc_std"],
                fmt="o", ms=8, color="#3979a8", ecolor="#577a91", capsize=5,
                linewidth=2)
    ax.set_yticks(y_pos, d["model"])
    ax.set_xlabel("Mean 5-fold cross-validation ROC-AUC (± 1 SD)")
    ax.set_xlim(max(0.0, float(d["cv_auc_mean"].min())-0.02), 1.008)
    ax.set_title("Model comparison: training folds only")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def plot_cm(metrics, out):
    matrix = np.array([[metrics["true_benign"], metrics["false_malignant"]],
                       [metrics["missed_malignant"], metrics["correct_malignant"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Untouched test: selected model")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if im.norm(value)>0.5 else "black", fontsize=16)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def plot_roc(y_test, test_scores, metrics, out):
    fpr, tpr, _ = roc_curve(y_test, test_scores, pos_label=1)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(fpr, tpr, lw=2, label=f"Selected model (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="gray", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate",
           ylabel="True-positive rate (sensitivity)", title="Held-out test ROC curve")
    ax.legend(loc="lower right"); ax.grid(alpha=0.22)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def plot_pca(X_train, out):
    """Separate descriptive PCA on train only; not used to select model."""
    X_std = StandardScaler().fit_transform(X_train)
    pca = PCA().fit(X_std)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(0.95, color="#c46e61", ls="--", label="95% threshold")
    ax.axvline(n95, color="#669a73", ls=":", label=f"{n95} components reach 95%")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
           title="PCA of standardized training features", ylim=(0, 1.03))
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)
    return n95


def run_experiment(output_dir):
    """End-to-end deterministic benchmark. Writes figures, tables and metadata."""
    root = Path(output_dir)
    results_dir = root / "results"; figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = get_data()
    tr, te = split_data(X, y)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    assert set(tr).isdisjoint(set(te))
    assert len(tr)+len(te)==len(y)

    splits = pd.DataFrame({"original_row_index": np.arange(len(X)), "split": "test"})
    splits.loc[tr, "split"] = "train"
    splits.to_csv(results_dir / "split_indices.csv", index=False)

    models = make_models()
    cv_table = cross_validate_models(models, X_tr, y_tr)
    cv_table.to_csv(results_dir / "cv_results.csv", index=False, float_format="%.8f")
    selected_name = select_model(cv_table)
    selected = models[selected_name]
    selected.fit(X_tr, y_tr)  # fitted only after cross-validation model selection
    metrics, preds, test_scores = eval_selected(selected, X_te, y_te)
    metrics["selected_model"] = selected_name
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(results_dir / "test_results.csv", index=False)
    pd.DataFrame({"original_row_index": te, "true_malignant": y_te.to_numpy(),
                  "predicted_malignant": preds, "model_decision_score": test_scores}).to_csv(
        results_dir / "test_predictions.csv", index=False, float_format="%.8f")

    n95 = plot_pca(X_tr, figures_dir / "pca_explained_variance.png")
    plot_dataset(y, figures_dir / "dataset_distribution.png")
    plot_cv(cv_table, figures_dir / "cross_validation_comparison.png")
    plot_cm(metrics, figures_dir / "test_confusion_matrix.png")
    plot_roc(y_te, test_scores, metrics, figures_dir / "test_roc_curve.png")
    summary = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn built-in)",
        "samples": int(len(X)), "features": int(X.shape[1]),
        "malignant_positive_class": 1, "malignant_samples": int(y.sum()),
        "benign_samples": int((y==0).sum()),
        "train_n": int(len(tr)), "test_n": int(len(te)),
        "train_malignant": int(y_tr.sum()), "test_malignant": int(y_te.sum()),
        "random_seed": SEED, "test_fraction": TEST_SIZE, "cv_folds": CV_FOLDS,
        "pca_train_components_95percent": n95, "selected_model": selected_name,
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "feature_scaling_and_PCA_within_CV_pipelines": True,
        "test_reserved_until_selection": True,
        "patient_level_splits_verified": False,
    }
    (results_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return cv_table, metrics, summary


if __name__ == "__main__":
    tables, metrics, summary = run_experiment(Path(__file__).resolve().parent.parent)
    print(tables.to_string(index=False))
    print("\nSelected:", summary["selected_model"])
    print("Final untouched-test results:", json.dumps(metrics, indent=2))
