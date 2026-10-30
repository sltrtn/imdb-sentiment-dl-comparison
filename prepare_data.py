"""Build the cleaned train/val/test split used by every model.

Source data: IMDB Large Movie Review Dataset (Maas et al., 2011), 50,000 labelled
reviews. We read it from the official Hugging Face mirror (`stanfordnlp/imdb`,
which is a byte-for-byte copy of the aclImdb release) and fall back to a local
`aclImdb/` directory if one is present.

The official training set (25k) is split 90/10 into train/val with a fixed seed;
the official 25k test set is kept untouched for final evaluation.

Usage:
    python3 prepare_data.py            # downloads/caches, writes splits.json
    python3 prepare_data.py --force    # rebuild splits.json from cache
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "aclImdb")
CACHE_DIR = os.path.join(ROOT, "data")
SPLITS_PATH = os.path.join(ROOT, "splits.json")

SEED = 42
VAL_FRACTION = 0.1
HF_BASE = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text"

TAG_RE = re.compile(r"<[^>]+>")


def clean_text(t):
    """Strip HTML tags, collapse all whitespace (same cleaning as the interim run)."""
    t = TAG_RE.sub(" ", t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_from_aclImdb(split):
    """Read the original aclImdb/<split>/{pos,neg}/*.txt layout."""
    texts, labels = [], []
    for label, folder in [(1, "pos"), (0, "neg")]:
        pattern = os.path.join(DATA_DIR, split, folder, "*.txt")
        for fp in sorted(glob.glob(pattern)):
            with open(fp, encoding="utf-8") as f:
                texts.append(clean_text(f.read()))
            labels.append(label)
    if not texts:
        raise FileNotFoundError(f"no .txt files under {DATA_DIR}/{split}")
    return texts, labels


def download_parquet(split):
    """Download (once) and return the HF parquet mirror for a split."""
    dest = os.path.join(CACHE_DIR, f"{split}.parquet")
    if not os.path.exists(dest):
        os.makedirs(CACHE_DIR, exist_ok=True)
        url = f"{HF_BASE}/{split}-00000-of-00001.parquet"
        print(f"downloading {url} -> {dest}")
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    return dest


def load_from_parquet(split):
    """Read the HF mirror; requires pyarrow (pip install pyarrow)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required to read the parquet mirror: pip install pyarrow")

    table = pq.read_table(download_parquet(split)).to_pandas()
    texts = [clean_text(t) for t in table["text"].tolist()]
    labels = [int(v) for v in table["label"].tolist()]
    return texts, labels


def load_split(split):
    if os.path.isdir(os.path.join(DATA_DIR, split)):
        return load_from_aclImdb(split)
    return load_from_parquet(split)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json")
    args = ap.parse_args()

    if os.path.exists(SPLITS_PATH) and not args.force:
        print(f"{SPLITS_PATH} already exists (use --force to rebuild)")
        return

    random.seed(SEED)
    train_texts, train_labels = load_split("train")
    test_texts, test_labels = load_split("test")

    # Shuffle once, then carve a 10% validation split off the official train set.
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    n_val = int(VAL_FRACTION * len(combined))
    val = combined[:n_val]
    tr = combined[n_val:]

    out = {
        "train": {"texts": [t for t, _ in tr], "labels": [y for _, y in tr]},
        "val": {"texts": [t for t, _ in val], "labels": [y for _, y in val]},
        "test": {"texts": test_texts, "labels": test_labels},
    }
    with open(SPLITS_PATH, "w") as f:
        json.dump(out, f)

    print("train:", len(tr), "val:", len(val), "test:", len(test_texts))
    print(f"train pos ratio: {sum(y for _, y in tr) / len(tr):.4f}")
    print(f"val   pos ratio: {sum(y for _, y in val) / len(val):.4f}")
    print(f"test  pos ratio: {sum(test_labels) / len(test_labels):.4f}")
    print("sample:", tr[0][0][:200])
    print(f"wrote {SPLITS_PATH}")


if __name__ == "__main__":
    main()
