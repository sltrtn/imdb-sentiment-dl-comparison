"""MLP + TF-IDF baseline for IMDB sentiment classification.

TF-IDF (unigrams + bigrams, 20k features, English stop-words removed) feeding a
one-hidden-layer scikit-learn MLPClassifier. A small grid over hidden-layer size
and L2 regularisation is selected on the validation split; the winner is evaluated
on the official test set.

Results -> mlp_results.json

Usage:
    python3 train_mlp.py
    python3 train_mlp.py --configs 0 1
"""

import argparse
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from common import binary_metrics, load_splits, save_json

# (name, hidden_layer_sizes, alpha)
CONFIGS = [
    ("mlp-128-a1e-4", (128,), 1e-4),
    ("mlp-256-a1e-4", (256,), 1e-4),
    ("mlp-128-a1e-3", (128,), 1e-3),
]


def train_config(name, hidden, alpha, X_tr, y_tr, X_val, y_val, X_test, y_test, seed):
    print(f"\n=== {name}: hidden={hidden} alpha={alpha} ===")
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        n_iter_no_change=3,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"trained in {train_time:.1f}s, iterations={clf.n_iter_}")

    val_metrics = binary_metrics(y_val, clf.predict(X_val), clf.predict_proba(X_val)[:, 1])
    test_metrics = binary_metrics(y_test, clf.predict(X_test), clf.predict_proba(X_test)[:, 1])
    print(f"[{name}] val  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"[{name}] test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")
    print(f"[{name}] confusion matrix (test):\n{confusion_matrix(y_test, clf.predict(X_test))}")

    n_params = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    return {
        "name": name,
        "config": {
            "hidden_layer_sizes": list(hidden),
            "alpha": alpha,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 128,
            "learning_rate_init": 1e-3,
            "max_iter": 30,
            "early_stopping": True,
            "features": "TF-IDF unigram+bigram, 20000 features, min_df=2, english stop-words",
        },
        "n_params": int(n_params),
        "n_iter": int(clf.n_iter_),
        "train_time_sec": train_time,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    train, val, test = load_splits()

    print("Fitting TF-IDF vectorizer...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X_tr = vectorizer.fit_transform(train["texts"])
    X_val = vectorizer.transform(val["texts"])
    X_test = vectorizer.transform(test["texts"])
    print(f"Vectorized in {time.time() - t0:.1f}s. Vocab size: {len(vectorizer.vocabulary_)}")

    indices = args.configs if args.configs is not None else range(len(CONFIGS))
    runs = [
        train_config(*CONFIGS[i], X_tr, train["labels"], X_val, val["labels"],
                     X_test, test["labels"], args.seed)
        for i in indices
    ]

    best = max(runs, key=lambda r: r["val"]["f1"])
    print(f"\nBest config: {best['name']} (val f1={best['val']['f1']:.4f})")

    save_json({
        "model": "MLP + TF-IDF",
        "selection_metric": "val_f1",
        "tuning": runs,
        "best": best["name"],
        "val": best["val"],
        "test": best["test"],
        "config": best["config"],
        "n_params": best["n_params"],
        "train_time_sec": best["train_time_sec"],
        "total_tuning_time_sec": sum(r["train_time_sec"] for r in runs),
        "n_iter": best["n_iter"],
        "vocab_size": len(vectorizer.vocabulary_),
        "seed": args.seed,
    }, "mlp_results.json")


if __name__ == "__main__":
    main()
