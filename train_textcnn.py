"""TextCNN sentiment classifier on IMDB.

Kim (2014) style architecture: 100-d embeddings -> parallel 1-D convolutions with
kernel sizes 3/4/5 -> global max-pooling -> dropout -> linear classifier.

The interim run used randomly initialised, trainable embeddings. This version
supports initialising the embedding matrix from GloVe (glove-wiki-gigaword-100)
and selects among a few configurations on the validation split. The winning
configuration is evaluated once on the official test set.

Results -> textcnn_results.json

Usage:
    python3 train_textcnn.py                 # full config search
    python3 train_textcnn.py --configs 0 1   # subset
"""

import argparse
import re
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (
    binary_metrics,
    count_parameters,
    embedding_matrix,
    get_device,
    load_glove,
    load_splits,
    save_json,
    set_seed,
)

TOKEN_RE = re.compile(r"[a-z']+")
PAD, UNK = 0, 1

# (name, embed_init, freeze_embeddings, max_len, num_filters, lr)
CONFIGS = [
    ("textcnn-rand-200", "random", False, 200, 100, 1e-3),
    ("textcnn-glove-200", "glove", False, 200, 100, 1e-3),
    ("textcnn-glove-300", "glove", False, 300, 128, 1e-3),
]


def tokenize(t):
    return TOKEN_RE.findall(t.lower())


def build_vocab(texts, vocab_size=20000, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    words = [w for w, c in counter.most_common() if c >= min_freq][: vocab_size - 2]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(text, word2idx, max_len):
    ids = [word2idx.get(tok, UNK) for tok in tokenize(text)][:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.X = [encode(t, word2idx, max_len) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.float32)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix=None, embed_dim=100, num_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (B, E, L)
        feats = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1))).squeeze(1)


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    scores, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            n += xb.size(0)
            scores.append(torch.sigmoid(logits).cpu())
            labels.append(yb.cpu())
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    return total_loss / n, (scores > 0.5).astype(int), labels, scores


def train_config(cfg, splits, glove, device, epochs, patience, seed):
    name, embed_init, freeze, max_len, num_filters, lr = cfg
    set_seed(seed)
    train, val, test = splits

    word2idx = build_vocab(train["texts"])
    emb = None
    coverage = 0.0
    if embed_init == "glove":
        emb, coverage = embedding_matrix(word2idx, glove, seed=seed)

    loaders = {}
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        ds = IMDBDataset(split["texts"], split["labels"], word2idx, max_len)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=64 if split_name == "train" else 256,
            shuffle=split_name == "train",
            generator=torch.Generator().manual_seed(seed) if split_name == "train" else None,
        )

    model = TextCNN(len(word2idx) + 2, emb, num_filters=num_filters, freeze_embeddings=freeze).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n=== {name}: init={embed_init} max_len={max_len} filters={num_filters} lr={lr} "
          f"freeze_embeddings={freeze} params={count_parameters(model):,} ===")
    best_f1, best_state, best_epoch, history = -1.0, None, 0, []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n = 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total_loss / n
        val_loss, val_pred, val_true, val_scores = run_eval(model, loaders["val"], criterion, device)
        val_metrics = binary_metrics(val_true, val_pred, val_scores)
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["f1"],
                        "epoch_time_sec": elapsed})
        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    val_loss, val_pred, val_true, val_scores = run_eval(model, loaders["val"], criterion, device)
    test_loss, test_pred, test_true, test_scores = run_eval(model, loaders["test"], criterion, device)
    val_metrics = binary_metrics(val_true, val_pred, val_scores)
    test_metrics = binary_metrics(test_true, test_pred, test_scores)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss
    print(f"[{name}] val  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"[{name}] test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")

    return {
        "name": name,
        "config": {
            "embed_init": embed_init,
            "freeze_embeddings": freeze,
            "max_len": max_len,
            "vocab_size": len(word2idx) + 2,
            "embed_dim": 100,
            "num_filters": num_filters,
            "kernel_sizes": [3, 4, 5],
            "dropout": 0.5,
            "lr": lr,
            "batch_size": 64,
            "glove_coverage": coverage,
            "embedding_type": "GloVe glove-wiki-gigaword-100 (fine-tuned)" if embed_init == "glove"
            else "trainable (random init)",
        },
        "n_params": count_parameters(model),
        "history": history,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    splits = tuple(load_splits())
    glove = load_glove()

    indices = args.configs if args.configs is not None else range(len(CONFIGS))
    runs = [train_config(CONFIGS[i], splits, glove, device, args.epochs, args.patience, args.seed)
            for i in indices]

    best = max(runs, key=lambda r: r["val"]["f1"])
    print(f"\nBest config: {best['name']} (val f1={best['val']['f1']:.4f})")

    save_json({
        "model": "TextCNN",
        "selection_metric": "val_f1",
        "tuning": runs,
        "best": best["name"],
        "val": best["val"],
        "test": best["test"],
        "config": best["config"],
        "n_params": best["n_params"],
        "history": best["history"],
        "best_epoch": best["best_epoch"],
        "total_train_time_sec": best["train_time_sec"],
        "total_tuning_time_sec": sum(r["train_time_sec"] for r in runs),
        "device": str(device),
        "seed": args.seed,
    }, "textcnn_results.json")


if __name__ == "__main__":
    main()
