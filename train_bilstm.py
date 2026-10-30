"""BiLSTM sentiment classifier on IMDB, with GloVe-initialised embeddings.

Pipeline: tokenise -> fixed 50k vocabulary -> 100-d GloVe (glove-wiki-gigaword-100)
initialised and fine-tuned embeddings -> 1-layer bidirectional LSTM -> concat of
final forward/backward states -> dropout -> linear classifier.

A small grid is evaluated on the validation split and the best configuration is
reported on the held-out test set. Results -> bilstm_results.json

Usage:
    python3 train_bilstm.py                # full grid (default)
    python3 train_bilstm.py --epochs 6 --configs 0   # single config, quick run
"""

import argparse
import re
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
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

# (name, hidden_dim, num_layers, dropout, lr, freeze_embeddings)
CONFIGS = [
    ("bilstm-128-frozen", 128, 1, 0.5, 1e-3, True),
    ("bilstm-128-finetune", 128, 1, 0.5, 1e-3, False),
    ("bilstm-256-finetune", 256, 1, 0.4, 5e-4, False),
]


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, max_size=50000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    words = [w for w, c in counter.most_common() if c >= min_freq][: max_size - 2]
    return {w: i + 2 for i, w in enumerate(words)}


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.X = [[word2idx.get(tok, UNK) for tok in tokenize(t)][:max_len] or [UNK] for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.5, freeze_embeddings=False):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers*2, B, hidden)
        out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(out)).squeeze(1)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    all_scores, all_labels = [], []
    torch.set_grad_enabled(training)
    for xb, lengths, yb in loader:
        xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
        logits = model(xb, lengths)
        loss = criterion(logits, yb)
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_scores.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(yb.detach().cpu())
    torch.set_grad_enabled(True)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / n, (scores > 0.5).astype(int), labels, scores


def train_config(cfg, data, glove, device, epochs, patience, max_len, seed):
    name, hidden, layers, dropout, lr, freeze = cfg
    set_seed(seed)
    train, val, test = data
    word2idx = build_vocab(train["texts"])
    emb_matrix, coverage = embedding_matrix(word2idx, glove, seed=seed)

    train_ds = IMDBDataset(train["texts"], train["labels"], word2idx, max_len)
    val_ds = IMDBDataset(val["texts"], val["labels"], word2idx, max_len)
    test_ds = IMDBDataset(test["texts"], test["labels"], word2idx, max_len)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=128, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=collate)

    model = BiLSTMClassifier(emb_matrix, hidden, layers, dropout, freeze).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n=== {name}: hidden={hidden} layers={layers} dropout={dropout} lr={lr} "
          f"freeze_embeddings={freeze} params={count_parameters(model):,} ===")
    best_f1, best_state, best_epoch = -1.0, None, 0
    history = []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_pred, val_true, val_scores = run_epoch(model, val_loader, criterion, device)
        val_f1 = binary_metrics(val_true, val_pred)["f1"]
        val_acc = float((val_pred == val_true).mean())
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": val_acc, "val_f1": val_f1, "epoch_time_sec": elapsed})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    val_loss, val_pred, val_true, val_scores = run_epoch(model, val_loader, criterion, device)
    test_loss, test_pred, test_true, test_scores = run_epoch(model, test_loader, criterion, device)
    val_metrics = binary_metrics(val_true, val_pred, val_scores)
    test_metrics = binary_metrics(test_true, test_pred, test_scores)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss
    print(f"[{name}] val  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} "
          f"auc={val_metrics['roc_auc']:.4f}")
    print(f"[{name}] test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
          f"auc={test_metrics['roc_auc']:.4f}")

    return {
        "name": name,
        "config": {
            "hidden_dim": hidden,
            "num_layers": layers,
            "dropout": dropout,
            "lr": lr,
            "freeze_embeddings": freeze,
            "batch_size": 64,
            "max_len": max_len,
            "vocab_size": len(word2idx) + 2,
            "embed_dim": 100,
            "embedding_type": "GloVe glove-wiki-gigaword-100 (fine-tuned)" if not freeze
            else "GloVe glove-wiki-gigaword-100 (frozen)",
            "glove_coverage": coverage,
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
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", type=int, nargs="*", default=None,
                    help="indices of CONFIGS to run (default: all)")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    data = tuple(load_splits())
    glove = load_glove()

    indices = args.configs if args.configs is not None else range(len(CONFIGS))
    runs = []
    for i in indices:
        runs.append(train_config(CONFIGS[i], data, glove, device, args.epochs,
                                 args.patience, args.max_len, args.seed))

    best = max(runs, key=lambda r: r["val"]["f1"])
    print(f"\nBest config: {best['name']} (val f1={best['val']['f1']:.4f})")

    save_json({
        "model": "BiLSTM",
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
    }, "bilstm_results.json")


if __name__ == "__main__":
    main()
