import json, time, os, re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)
np.random.seed(42)

with open(os.path.join(ROOT, "splits.json")) as f:
    data = json.load(f)

tr_texts, tr_labels = data["train"]["texts"], data["train"]["labels"]
val_texts, val_labels = data["val"]["texts"], data["val"]["labels"]
test_texts, test_labels = data["test"]["texts"], data["test"]["labels"]

TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(t):
    return TOKEN_RE.findall(t.lower())

print("Building vocabulary...")
counter = Counter()
for t in tr_texts:
    counter.update(tokenize(t))

VOCAB_SIZE = 20000
MAX_LEN = 200
most_common = counter.most_common(VOCAB_SIZE - 2)  # reserve 0=pad, 1=unk
word2idx = {w: i + 2 for i, (w, _) in enumerate(most_common)}
PAD, UNK = 0, 1

def encode(text):
    ids = [word2idx.get(tok, UNK) for tok in tokenize(text)]
    ids = ids[:MAX_LEN]
    ids = ids + [PAD] * (MAX_LEN - len(ids))
    return ids

class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [encode(t) for t in texts]
        self.y = labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

print("Encoding splits...")
train_ds = IMDBDataset(tr_texts, tr_labels)
val_ds = IMDBDataset(val_texts, val_labels)
test_ds = IMDBDataset(test_texts, test_labels)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (B, embed_dim, L)
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(emb))
            pooled = torch.max(c, dim=2).values
            feats.append(pooled)
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.fc(out).squeeze(1)

device = torch.device("cpu")
model = TextCNN(vocab_size=VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

def run_eval(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    avg_loss = total_loss / len(all_labels)
    return np.array(all_preds), np.array(all_labels), avg_loss

EPOCHS = 4
history = []
t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    train_loss = total_loss / n
    val_preds, val_labels_arr, val_loss = run_eval(val_loader)
    val_acc = (val_preds == val_labels_arr).mean()
    epoch_time = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s")
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "epoch_time_sec": epoch_time})

total_train_time = time.time() - t_start

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def full_eval(loader, name):
    preds, labels_arr, loss = run_eval(loader)
    acc = accuracy_score(labels_arr, preds)
    prec = precision_score(labels_arr, preds)
    rec = recall_score(labels_arr, preds)
    f1 = f1_score(labels_arr, preds)
    cm = confusion_matrix(labels_arr, preds)
    print(f"[{name}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} loss={loss:.4f}")
    print(f"[{name}] confusion matrix:\n{cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "loss": loss, "confusion_matrix": cm.tolist()}

results = {
    "history": history,
    "val": full_eval(val_loader, "val"),
    "test": full_eval(test_loader, "test"),
    "total_train_time_sec": total_train_time,
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "embed_dim": 100,
    "num_filters": 100,
    "kernel_sizes": [3, 4, 5],
    "epochs": EPOCHS,
    "embedding_type": "trainable (random init, NOT pretrained GloVe)",
}

with open(os.path.join(ROOT, "textcnn_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Saved textcnn_results.json")
print(f"Total train time: {total_train_time:.1f}s")
