# Emotion API

Small CPU-only English emotion classifier with a FastAPI service.
Labels: `joy, sadness, fear, anger, guilt, motivation, neutral`.

`POST /predict` returns every emotion whose confidence is at or above the tuned threshold (0.25),
and always at least the top 1. Confidence is the model probability as a percentage, so the values
for several labels can add up to more or less than 100.

## Run the API

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

The model file `model.joblib` was saved with scikit-learn 1.9.1 (Python 3.14). Use the same
scikit-learn version (pinned in `requirements.txt`) or loading may fail.

```bash
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"text": "I am so happy I got the job"}'
# {"emotions":[{"label":"joy","confidence":98.8}]}
```

Node.js:

```js
const res = await fetch("http://localhost:8001/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "I am so happy I got the job" }),
});
const { emotions } = await res.json(); // [{ label, confidence }]
```

`GET /health` returns the loaded model and labels. Empty text returns HTTP 422.
Set `MODEL_PATH` to load a different model file.

## Model

- TF-IDF (word 1-2 grams) + Logistic Regression (balanced class weights). No GPU, no torch.
- A MiniLM-embedding + Logistic Regression model was also trained and was worse (macro-F1 0.51 vs 0.64), so it was dropped.
- Training data: reference repo `FaheemRafiq/emotion-classifier` (GoEmotions + ISEAR + mixed-emotion rows),
  mapped to the 7 labels. Data is not stored here.

## Metrics

| Set | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation (GoEmotions) | 3,359 | 0.73 | 0.63 |
| **Frozen test (GoEmotions)** | 3,400 | 0.73 | 0.64 |
| 154 check-ins (synthetic, unreviewed) | 154 | 0.48 | 0.49 |

Per-emotion F1 on the frozen test split: joy 0.86, neutral 0.77, fear 0.62, motivation 0.57,
sadness 0.57, anger 0.55, guilt 0.52. Full numbers are in `metrics.json`.

## Known limits (read before relying on it)

- Trained on Reddit comments, not journal check-ins. On the 154 check-ins accuracy drops to 0.48.
  Those 154 rows are AI-written and their labels were never reviewed, so treat that number as a rough signal.
- `guilt` and `motivation` are weaker: GoEmotions has no such labels, so they are proxies
  (guilt from remorse/embarrassment plus ISEAR guilt/shame; motivation from optimism/desire/pride).
- Short or imperative sentences ("I will finish this today") are often predicted as neutral.
- Best next step for quality: add a few hundred reviewed check-in style examples per emotion and retrain.

## Retrain

```bash
pip install -r requirements-train.txt
git clone https://github.com/FaheemRafiq/emotion-classifier
(cd emotion-classifier && python -m src.prepare_data && python -m src.prepare_extra)
python train.py --repo ./emotion-classifier      # add --skip-embed to train only the TF-IDF model
```
