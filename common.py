"""Shared helpers: data loading, seeding, GloVe loading, metrics, result serialisation."""

import gzip
import json
import os
import random

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS_PATH = os.path.join(ROOT, "splits.json")
SEED = 42
GLOVE_CACHE = os.path.expanduser("~/.cache/glove/glove-wiki-gigaword-100.gz")
GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_splits():
    """Return (train, val, test) where each split is a dict with texts/labels."""
    with open(SPLITS_PATH) as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]


def binary_metrics(y_true, y_pred, scores=None):
    """Standard binary classification metrics, JSON-serialisable."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if scores is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    return out


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_json(obj, name):
    path = os.path.join(ROOT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {name}")
    return path


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_glove(path=GLOVE_CACHE, url=GLOVE_URL):
    """Load 100-d GloVe vectors as {word: np.ndarray}, downloading once if needed."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import torch

        print(f"Downloading GloVe vectors -> {path}")
        torch.hub.download_url_to_file(url, path, progress=True)
    vectors = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != 101:
                continue
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    print(f"Loaded {len(vectors)} GloVe vectors from {path}")
    return vectors


def embedding_matrix(word2idx, glove, embed_dim=100, seed=SEED, pad_index=0):
    """Random-init embedding matrix with GloVe rows where available; returns (matrix, coverage)."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(word2idx) + 2, embed_dim)).astype(np.float32)
    matrix[pad_index] = 0.0
    hits = 0
    for word, idx in word2idx.items():
        vec = glove.get(word)
        if vec is not None:
            matrix[idx] = vec
            hits += 1
    coverage = hits / max(len(word2idx), 1)
    print(f"GloVe coverage: {hits}/{len(word2idx)} ({coverage:.1%}) of vocabulary")
    return matrix, coverage
