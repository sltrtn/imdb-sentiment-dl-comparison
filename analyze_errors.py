"""Qualitative + sliced error analysis for the fine-tuned DistilBERT model.

Runs the best DistilBERT checkpoint over the official test set and reports:
  * accuracy by review length (word-count quartiles)
  * accuracy on reviews containing negation cues ("not", "n't", "never", "no")
  * the most confident false positives / false negatives, with snippets

Outputs results/error_analysis.md and results/figures/accuracy_by_length.png.

Usage:
    python3 analyze_errors.py
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, get_device, load_splits

RESULTS_PATH = os.path.join(ROOT, "distilbert_results.json")
OUT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")
NEGATION_RE = re.compile(r"\b(not|no|never|n't|cannot|hardly)\b", re.IGNORECASE)


class TestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def snippet(text, limit=340):
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    model_dir = os.path.join(ROOT, "models", results["best"])
    if not os.path.isdir(model_dir):
        raise SystemExit(f"checkpoint not found at {model_dir}; run train_distilbert.py first")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    _, _, test = load_splits()
    max_len = results["config"]["max_len"]
    enc = tokenizer(test["texts"], padding="max_length", truncation=True, max_length=max_len,
                    return_tensors="pt")
    loader = DataLoader(TestDataset(enc, test["labels"]), batch_size=64)
    probs = predict(model, loader, device)
    preds = (probs > 0.5).astype(int)
    labels = np.asarray(test["labels"])
    correct = preds == labels
    print(f"test accuracy: {correct.mean():.4f}")

    # --- accuracy by review length (word counts) ---
    lengths = np.array([len(t.split()) for t in test["texts"]])
    edges = np.quantile(lengths, [0, 0.25, 0.5, 0.75, 1.0]).astype(int)
    edges[-1] += 1
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        buckets.append({
            "range": f"{lo}–{hi - 1}",
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
        })
        print(f"  length {lo:4d}-{hi - 1:4d} words: n={mask.sum():5d} acc={correct[mask].mean():.4f}")

    # --- negation slice ---
    neg_mask = np.array([bool(NEGATION_RE.search(t)) for t in test["texts"]])
    slices = {
        "with negation cue": {
            "n": int(neg_mask.sum()),
            "accuracy": float(correct[neg_mask].mean()),
            "error_rate": float(1 - correct[neg_mask].mean()),
        },
        "without negation cue": {
            "n": int((~neg_mask).sum()),
            "accuracy": float(correct[~neg_mask].mean()),
            "error_rate": float(1 - correct[~neg_mask].mean()),
        },
    }
    print("  negation slice:", slices)

    # --- most confident mistakes ---
    errors = []
    for i in np.where(~correct)[0]:
        confidence = probs[i] if preds[i] == 1 else 1 - probs[i]
        errors.append({
            "index": int(i),
            "true": "positive" if labels[i] == 1 else "negative",
            "predicted": "positive" if preds[i] == 1 else "negative",
            "confidence": float(confidence),
            "words": int(lengths[i]),
            "text": snippet(test["texts"][i]),
        })
    errors.sort(key=lambda e: e["confidence"], reverse=True)
    false_pos = [e for e in errors if e["predicted"] == "positive"][:5]
    false_neg = [e for e in errors if e["predicted"] == "negative"][:5]

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = range(len(buckets))
    ax.bar(xs, [b["accuracy"] for b in buckets], color="#C44E52")
    ax.set_xticks(list(xs), [f"{b['range']}\n(n={b['n']:,})" for b in buckets], fontsize=8)
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel("Review length (words, test-set quartiles)")
    ax.set_ylabel("Accuracy")
    ax.set_title("DistilBERT accuracy by review length")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "accuracy_by_length.png"), dpi=160)
    plt.close(fig)

    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [
        "# DistilBERT error analysis\n",
        f"Checkpoint: `models/{results['best']}` — test accuracy **{correct.mean():.4f}** "
        f"({int(correct.sum()):,}/{len(correct):,})\n",
        "## Accuracy by review length\n",
        "| Words | Reviews | Accuracy |",
        "|---|---|---|",
    ]
    lines += [f"| {b['range']} | {b['n']:,} | {b['accuracy']:.4f} |" for b in buckets]
    lines += [
        "\n## Negation slice\n",
        "| Slice | Reviews | Accuracy | Error rate |",
        "|---|---|---|---|",
    ]
    lines += [f"| {k} | {v['n']:,} | {v['accuracy']:.4f} | {v['error_rate']:.4f} |"
              for k, v in slices.items()]
    lines += ["\n## Most confident false positives\n"]
    for e in false_pos:
        lines += [f"- **confidence {e['confidence']:.3f}**, {e['words']} words — “{e['text']}”"]
    lines += ["\n## Most confident false negatives\n"]
    for e in false_neg:
        lines += [f"- **confidence {e['confidence']:.3f}**, {e['words']} words — “{e['text']}”"]
    with open(os.path.join(OUT_DIR, "error_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join('results', 'error_analysis.md')}")


if __name__ == "__main__":
    main()
