"""Train a small CPU-only emotion classifier (7 labels) and save model.joblib.

Compares:  A) TF-IDF + LogisticRegression   B) MiniLM embeddings + LogisticRegression
Keeps the better one (by validation macro-F1), tunes the multi-label threshold,
and reports metrics on validation, the frozen test split and the 154 mentor check-ins.

Usage (after running the reference repo's data scripts):
  python train.py --repo ./emotion-classifier
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

# Output names in the API. Repo uses fear_anxiety / anger_frustration internally.
RENAME = {"fear_anxiety": "fear", "anger_frustration": "anger"}
LABELS = ["joy", "sadness", "fear", "anger", "guilt", "motivation", "neutral"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_labels(s):
    return s.map(lambda x: RENAME.get(x, x))


def load_data(repo: Path):
    p3, p2 = repo / "data/processed_v3", repo / "data/processed"
    train = pd.read_parquet(p3 / "train.parquet")
    train = train[train["label"].notna()]  # hard-label rows only (keeps it simple)
    val_path = p3 / "validation.parquet" if (p3 / "validation.parquet").exists() else p2 / "validation.parquet"
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(p2 / "test.parquet")  # frozen test split
    out = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        out[name] = (df["text"].tolist(), clean_labels(df["label"]).tolist())
    ev = pd.read_csv(repo / "data/mentor_eval/mentor_eval.csv")
    ev["label"] = clean_labels(ev["label"])
    ev["note"] = ev["note"].fillna("")
    out["mentor_all"] = (ev["text"].tolist(), ev["label"].tolist())
    clean = ev[ev["note"].str.strip() == ""]
    out["mentor_clean"] = (clean["text"].tolist(), clean["label"].tolist())
    return out


class Embedder:
    """Sentence embeddings (MiniLM). Loaded lazily so model B is optional."""

    def __init__(self, name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.m = SentenceTransformer(name, device="cpu")

    def __call__(self, texts):
        return self.m.encode(list(texts), batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True)


def sample_f1(proba, classes, y_true, thr):
    """Mean per-sample F1 between predicted label set and the single true label."""
    scores = []
    for p, y in zip(proba, y_true):
        pred = {classes[i] for i in np.where(p >= thr)[0]} or {classes[int(p.argmax())]}
        hit = y in pred
        scores.append(2 * hit / (len(pred) + 1))
    return float(np.mean(scores))


def tune_threshold(proba, classes, y_true):
    grid = np.arange(0.20, 0.71, 0.05)
    best = max(grid, key=lambda t: sample_f1(proba, classes, y_true, t))
    return round(float(best), 2)


def report(name, clf, X, y, classes, thr):
    proba = clf.predict_proba(X)
    pred = [classes[i] for i in proba.argmax(1)]
    row = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=classes, zero_division=0),
        "sample_f1_multilabel": sample_f1(proba, classes, y, thr),
    }
    per = f1_score(y, pred, average=None, labels=classes, zero_division=0)
    row.update({f"f1_{c}": float(v) for c, v in zip(classes, per)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="./emotion-classifier")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--skip-embed", action="store_true", help="only train model A")
    args = ap.parse_args()

    d = load_data(Path(args.repo))
    Xtr, ytr = d["train"]
    print({k: len(v[0]) for k, v in d.items()})

    candidates = {}
    # A) TF-IDF + LR
    a = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=5, max_iter=1000, class_weight="balanced"),
    )
    a.fit(Xtr, ytr)
    candidates["tfidf_lr"] = (a, None)

    # B) MiniLM embeddings + LR
    if not args.skip_embed:
        emb = Embedder()
        Etr = emb(Xtr)
        b = LogisticRegression(C=2, max_iter=2000, class_weight="balanced")
        b.fit(Etr, ytr)
        candidates["minilm_lr"] = (b, emb)

    def feats(model, emb, X):
        return emb(X) if emb is not None else X

    results, best_name, best_f1 = {}, None, -1
    for name, (clf, emb) in candidates.items():
        classes = list(clf.classes_)
        Xv, yv = d["val"]
        thr = tune_threshold(clf.predict_proba(feats(clf, emb, Xv)), classes, yv)
        results[name] = {"threshold": thr}
        for split in ["val", "test", "mentor_all", "mentor_clean"]:
            X, y = d[split]
            results[name][split] = report(name, clf, feats(clf, emb, X), y, classes, thr)
        f1 = results[name]["val"]["macro_f1"]
        print(f"{name}: threshold={thr} val macro-F1={f1:.3f} test macro-F1={results[name]['test']['macro_f1']:.3f}")
        if f1 > best_f1:
            best_name, best_f1 = name, f1

    clf, emb = candidates[best_name]
    classes = list(clf.classes_)
    assert set(classes) == set(LABELS), classes
    bundle = {
        "kind": best_name,
        "classifier": clf,
        "classes": classes,
        "threshold": results[best_name]["threshold"],
        "embed_model": EMBED_MODEL if emb is not None else None,
    }
    joblib.dump(bundle, args.out, compress=3)
    Path("metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {args.out} (best = {best_name}). Full metrics in metrics.json")


if __name__ == "__main__":
    main()