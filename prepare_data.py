import os, re, json, random, glob

random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "aclImdb")

TAG_RE = re.compile(r"<[^>]+>")

def clean_text(t):
    t = TAG_RE.sub(" ", t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_split(split):
    texts, labels = [], []
    for label, folder in [(1, "pos"), (0, "neg")]:
        path = os.path.join(DATA_DIR, split, folder, "*.txt")
        for fp in sorted(glob.glob(path)):
            with open(fp, encoding="utf-8") as f:
                texts.append(clean_text(f.read()))
            labels.append(label)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)

if __name__ == "__main__":
    train_texts, train_labels = load_split("train")
    test_texts, test_labels = load_split("test")

    # carve a validation set out of train (90/10), stratified by shuffling already random
    n_val = int(0.1 * len(train_texts))
    val_texts, val_labels = train_texts[:n_val], train_labels[:n_val]
    tr_texts, tr_labels = train_texts[n_val:], train_labels[n_val:]

    out = {
        "train": {"texts": tr_texts, "labels": tr_labels},
        "val": {"texts": val_texts, "labels": val_labels},
        "test": {"texts": test_texts, "labels": test_labels},
    }
    with open(os.path.join(ROOT, "splits.json"), "w") as f:
        json.dump(out, f)

    print("train:", len(tr_texts), "val:", len(val_texts), "test:", len(test_texts))
    print("train pos ratio:", sum(tr_labels)/len(tr_labels))
    print("val pos ratio:", sum(val_labels)/len(val_labels))
    print("test pos ratio:", sum(test_labels)/len(test_labels))
    print("sample:", tr_texts[0][:200])
