#!/usr/bin/env bash
# Reproduce the full study end to end.
#   ./run_all.sh            # all models
#   SKIP_DISTILBERT=1 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

echo "== 1/6 preparing data =="
$PY prepare_data.py

echo "== 2/6 MLP + TF-IDF =="
$PY train_mlp.py

echo "== 3/6 TextCNN =="
$PY train_textcnn.py

echo "== 4/6 BiLSTM =="
$PY train_bilstm.py

if [[ "${SKIP_DISTILBERT:-0}" != "1" ]]; then
  echo "== 5/6 DistilBERT =="
  $PY train_distilbert.py
fi

echo "== 6/6 comparing =="
$PY compare.py

echo "done — see results/ and README.md"
