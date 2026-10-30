"""Fine-tune DistilBERT for IMDB sentiment classification.

DistilBERT-base-uncased (66M params) with a sequence-classification head, fully
fine-tuned end to end. Two learning rates are compared on the validation split;
the best checkpoint (by validation F1) is evaluated once on the official test set.

Results -> distilbert_results.json

Usage:
    python3 train_distilbert.py
    python3 train_distilbert.py --epochs 2 --max-len 256 --lrs 2e-5
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ROOT, binary_metrics, get_device, load_splits, save_json, set_seed

MODEL_NAME = "distilbert-base-uncased"
CACHE_DIR = os.path.join(ROOT, ".hf_cache")


class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def tokenize_split(tokenizer, texts, max_len):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def evaluate(model, loader, device, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
            logits = out.logits.float()
            total_loss += out.loss.item() * batch["labels"].size(0)
            n += batch["labels"].size(0)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels.append(batch["labels"].cpu())
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    return total_loss / n, (scores > 0.5).astype(int), labels, scores


def train_config(lr, splits, tokenizer, device, args):
    set_seed(args.seed)
    train, val, test = splits
    train_enc = tokenize_split(tokenizer, train["texts"], args.max_len)
    val_enc = tokenize_split(tokenizer, val["texts"], args.max_len)
    test_enc = tokenize_split(tokenizer, test["texts"], args.max_len)

    train_loader = DataLoader(
        IMDBDataset(train_enc, train["labels"]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(IMDBDataset(val_enc, val["labels"]), batch_size=args.eval_batch_size)
    test_loader = DataLoader(IMDBDataset(test_enc, test["labels"]), batch_size=args.eval_batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, cache_dir=CACHE_DIR
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    use_amp = device.type == "cuda"

    print(f"\n=== distilbert lr={lr:g} max_len={args.max_len} batch={args.batch_size} "
          f"epochs={args.epochs} amp={use_amp} ===")
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    best_f1, best_state, best_epoch, history = -1.0, None, 0, []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += out.loss.item() * batch["labels"].size(0)
            n += batch["labels"].size(0)
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss={running_loss / n:.4f} "
                      f"({time.time() - t0:.0f}s)")
        train_loss = running_loss / n
        val_loss, val_pred, val_true, val_scores = evaluate(model, val_loader, device, use_amp)
        val_metrics = binary_metrics(val_true, val_pred, val_scores)
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} ({elapsed:.0f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["f1"],
                        "epoch_time_sec": elapsed})
        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    model.to(device)
    val_loss, val_pred, val_true, val_scores = evaluate(model, val_loader, device, use_amp)
    test_loss, test_pred, test_true, test_scores = evaluate(model, test_loader, device, use_amp)
    val_metrics = binary_metrics(val_true, val_pred, val_scores)
    test_metrics = binary_metrics(test_true, test_pred, test_scores)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss

    best_dir = os.path.join(ROOT, "models", f"distilbert-lr{lr:g}")
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"saved best checkpoint -> {best_dir}")
    print(f"[distilbert lr={lr:g}] val  acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"[distilbert lr={lr:g}] test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": f"distilbert-lr{lr:g}",
        "config": {
            "model_name": MODEL_NAME,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "lr": lr,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "epochs": args.epochs,
            "pretrained": "distilbert-base-uncased (fine-tuned)",
        },
        "n_params": int(n_params),
        "history": history,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lrs", type=float, nargs="*", default=[2e-5, 3e-5])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    set_seed(args.seed)
    splits = tuple(load_splits())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    runs = [train_config(lr, splits, tokenizer, device, args) for lr in args.lrs]
    best = max(runs, key=lambda r: r["val"]["f1"])
    print(f"\nBest config: {best['name']} (val f1={best['val']['f1']:.4f})")

    save_json({
        "model": "DistilBERT",
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
    }, "distilbert_results.json")


if __name__ == "__main__":
    main()
