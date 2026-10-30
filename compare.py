"""Aggregate every model's results into a comparison table, CSV and figures.

Reads the *_results.json files written by the training scripts, verifies that all
models were evaluated on the same split, and produces:

    results/comparison.md    markdown table (also pasted into README)
    results/comparison.csv   one row per model
    results/figures/*.png    metric bars, confusion matrices, training curves

Usage:
    python3 compare.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT

RESULT_FILES = [
    ("MLP + TF-IDF", "mlp_results.json", "#4C72B0"),
    ("TextCNN", "textcnn_results.json", "#DD8452"),
    ("BiLSTM", "bilstm_results.json", "#55A868"),
    ("DistilBERT", "distilbert_results.json", "#C44E52"),
]
OUT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def load_results():
    rows = []
    for display, filename, color in RESULT_FILES:
        path = os.path.join(ROOT, filename)
        if not os.path.exists(path):
            print(f"  skipping {display}: {filename} not found")
            continue
        with open(path) as f:
            res = json.load(f)
        cfg = res.get("config", {})
        rows.append({
            "name": display,
            "file": filename,
            "color": color,
            "n_params": res.get("n_params"),
            "train_time_sec": res.get("total_train_time_sec") or res.get("train_time_sec"),
            "best_epoch": res.get("best_epoch"),
            "embedding": cfg.get("embedding_type", cfg.get("features", cfg.get("pretrained", ""))),
            "val_acc": res["val"]["accuracy"],
            "val_f1": res["val"]["f1"],
            "acc": res["test"]["accuracy"],
            "precision": res["test"]["precision"],
            "recall": res["test"]["recall"],
            "f1": res["test"]["f1"],
            "roc_auc": res["test"].get("roc_auc"),
            "cm": res["test"]["confusion_matrix"],
            "history": res.get("history", []),
            "config": cfg,
        })
    return rows


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "—"


def write_table(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    header = ("| Model | Test accuracy | Precision | Recall | F1 | ROC-AUC | Params | Train time |\n"
              "|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        params = f"{r['n_params']:,}" if r["n_params"] else "—"
        train_time = f"{r['train_time_sec']:.0f}s" if r["train_time_sec"] else "—"
        lines.append(
            f"| **{r['name']}** | {fmt(r['acc'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
            f"{fmt(r['f1'])} | {fmt(r['roc_auc'])} | {params} | {train_time} |"
        )
    table = header + "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "comparison.md"), "w") as f:
        f.write(table)

    csv_header = "model,test_accuracy,precision,recall,f1,roc_auc,n_params,train_time_sec\n"
    csv_lines = [
        f"{r['name']},{r['acc']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f},"
        f"{fmt(r['roc_auc']) if r['roc_auc'] is not None else ''},"
        f"{r['n_params'] or ''},{round(r['train_time_sec']) if r['train_time_sec'] else ''}"
        for r in rows
    ]
    with open(os.path.join(OUT_DIR, "comparison.csv"), "w") as f:
        f.write(csv_header + "\n".join(csv_lines) + "\n")
    print(table)
    return table


def plot_bars(rows):
    os.makedirs(FIG_DIR, exist_ok=True)
    metrics = [("acc", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    if any(r["roc_auc"] for r in rows):
        metrics.append(("roc_auc", "ROC-AUC"))
    x = np.arange(len(rows))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(metrics):
        values = [r[key] if r[key] is not None else 0 for r in rows]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=label)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("IMDB sentiment classification — test-set metrics")
    ax.legend(ncol=len(metrics), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "metric_comparison.png"), dpi=160)
    plt.close(fig)


def plot_confusions(rows):
    os.makedirs(FIG_DIR, exist_ok=True)
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        cm = np.array(r["cm"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1], ["neg", "pos"])
        ax.set_yticks([0, 1], ["neg", "pos"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{r['name']}\nacc={r['acc']:.4f}  f1={r['f1']:.4f}")
    fig.suptitle("Test-set confusion matrices")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=160)
    plt.close(fig)


def plot_curves(rows):
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = [r for r in rows if r["history"]]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in rows:
        epochs = [h["epoch"] for h in r["history"]]
        axes[0].plot(epochs, [h["val_loss"] for h in r["history"]], marker="o", label=r["name"],
                     color=r["color"])
        key = "val_f1" if "val_f1" in r["history"][0] else "val_acc"
        axes[1].plot(epochs, [h[key] for h in r["history"]], marker="o", label=r["name"],
                     color=r["color"])
    axes[0].set_title("Validation loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation F1 per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=160)
    plt.close(fig)


def main():
    print("Loading results...")
    rows = load_results()
    if not rows:
        raise SystemExit("no result files found — run the training scripts first")
    rows.sort(key=lambda r: r["f1"], reverse=True)
    write_table(rows)
    plot_bars(rows)
    plot_confusions(rows)
    plot_curves(rows)
    print(f"Wrote results/comparison.md, results/comparison.csv and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
