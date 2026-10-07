"""FastAPI service: POST /predict {"text": "..."} -> emotions with confidence %."""
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
state = {}


@asynccontextmanager
async def lifespan(app):
    bundle = joblib.load(MODEL_PATH)
    state["clf"] = bundle["classifier"]
    state["classes"] = bundle["classes"]
    state["threshold"] = bundle["threshold"]
    state["embed"] = None
    if bundle.get("embed_model"):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(bundle["embed_model"], device="cpu")
        state["embed"] = lambda t: m.encode(t, normalize_embeddings=True)
    state["kind"] = bundle["kind"]
    yield


app = FastAPI(title="Emotion API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "model": state.get("kind"), "labels": state.get("classes")}


@app.post("/predict")
def predict(body: PredictIn):
    text = body.text.strip()
    if not text:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="text must not be empty")
    X = state["embed"]([text]) if state["embed"] else [text]
    proba = state["clf"].predict_proba(X)[0]
    classes, thr = state["classes"], state["threshold"]
    idx = [i for i in np.argsort(proba)[::-1] if proba[i] >= thr] or [int(proba.argmax())]
    return {"emotions": [{"label": classes[i], "confidence": round(float(proba[i]) * 100, 1)} for i in idx]}
