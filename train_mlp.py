import json, time, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT, "splits.json")) as f:
    data = json.load(f)

tr_texts, tr_labels = data["train"]["texts"], data["train"]["labels"]
val_texts, val_labels = data["val"]["texts"], data["val"]["labels"]
test_texts, test_labels = data["test"]["texts"], data["test"]["labels"]

print("Fitting TF-IDF vectorizer...")
t0 = time.time()
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
X_tr = vectorizer.fit_transform(tr_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)
print(f"Vectorized in {time.time()-t0:.1f}s. Vocab size: {len(vectorizer.vocabulary_)}")

print("Training MLP...")
t0 = time.time()
clf = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam",
                     alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                     max_iter=30, early_stopping=True, n_iter_no_change=3,
                     validation_fraction=0.1, random_state=42, verbose=True)
clf.fit(X_tr, tr_labels)
print(f"Trained in {time.time()-t0:.1f}s, iterations: {clf.n_iter_}")

def evaluate(X, y, name):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)
    print(f"[{name}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
    print(f"[{name}] confusion matrix:\n{cm}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm.tolist()}

results = {}
results["val"] = evaluate(X_val, val_labels, "val")
results["test"] = evaluate(X_test, test_labels, "test")
results["train_time_sec"] = time.time() - t0
results["n_iter"] = int(clf.n_iter_)
results["vocab_size"] = len(vectorizer.vocabulary_)

with open(os.path.join(ROOT, "mlp_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Saved mlp_results.json")
