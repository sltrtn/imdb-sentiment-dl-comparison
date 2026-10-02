# Comparative Analysis of Deep Learning Architectures for Sentiment Classification of Movie Reviews

Deep Learning course project (ICT-4442), Manipal Institute of Technology.

Team: Hasil Haroon (230953019), T Rama Mohan Aniketh (230911224), Madhavan K (230953298), Ayrton Pulikottil (230953448).

## Project

Trains and compares four deep learning architectures for binary sentiment classification
(positive/negative) on the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/):

1. **MLP + TF-IDF** (baseline) — Hasil Haroon
2. **TextCNN** — T Rama Mohan Aniketh
3. **BiLSTM** — Madhavan K (not yet implemented)
4. **Fine-tuned DistilBERT** — Ayrton Pulikottil (not yet implemented)

See the Interim Report (Part B) for full details, literature review, and preliminary results.

## Status (interim)

| Model | Status | Test accuracy | Test F1 |
|---|---|---|---|
| MLP + TF-IDF | Done | 0.8755 | 0.8761 |
| TextCNN | Done (trainable embeddings, GloVe pending) | 0.8485 | 0.8464 |
| BiLSTM | Not started | — | — |
| DistilBERT | Not started | — | — |

## Reproducing

```bash
# 1. Download and extract the dataset
curl -L -o aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzf aclImdb_v1.tar.gz

# 2. Build the cleaned train/val/test split
python3 prepare_data.py

# 3. Train each model
python3 train_mlp.py
python3 train_textcnn.py
```

`prepare_data.py` strips HTML tags, collapses whitespace, and carves a 10% validation
split out of the official training set (fixed seed 42), leaving the official test set
untouched. Results are written to `mlp_results.json` / `textcnn_results.json`.

## Known gaps (tracked for the Final Report)

- TextCNN currently uses randomly-initialized, trainable embeddings rather than pretrained
  GloVe vectors as originally proposed.
- BiLSTM and DistilBERT are not yet implemented.
- No hyperparameter tuning has been done yet; numbers above are a first pass.
