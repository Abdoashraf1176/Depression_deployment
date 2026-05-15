from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import os

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────

# If you have a fine-tuned model folder, set this env var to its path.
# Otherwise, falls back to the base mental-roberta-base from HuggingFace.
MODEL_PATH = os.environ.get("MODEL_PATH", "abdo1176/mental-roberta-finetuned")

DEVICE = "cpu"  # Railway free tier has no GPU

# ─────────────────────────────────────────
#  LOAD MODEL ON STARTUP
# ─────────────────────────────────────────
print(f"Loading model '{MODEL_PATH}' on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Model loaded successfully!")

LABEL_MAP = {i: label for i, label in enumerate(model.config.id2label.values())}
print(f"Labels: {LABEL_MAP}")

# ─────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title="MentalRoBERTa API",
    description="Fine-tuned Mental Health Text Classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

class PredictionResult(BaseModel):
    label: str
    label_id: int
    confidence: float
    all_scores: dict
    inference_time_ms: float

# ─────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────
def predict(text: str) -> PredictionResult:
    start = time.time()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = probs.argmax().item()
    pred_label = LABEL_MAP.get(pred_id, str(pred_id))
    confidence = probs[pred_id].item()
    all_scores = {LABEL_MAP.get(i, str(i)): round(p.item(), 4) for i, p in enumerate(probs)}
    elapsed_ms = (time.time() - start) * 1000

    return PredictionResult(
        label=pred_label,
        label_id=pred_id,
        confidence=round(confidence, 4),
        all_scores=all_scores,
        inference_time_ms=round(elapsed_ms, 2)
    )

# ─────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_PATH,
        "device": DEVICE,
        "labels": LABEL_MAP
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict", response_model=PredictionResult)
def predict_single(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return predict(body.text)

@app.post("/predict/batch")
def predict_batch(body: BatchInput):
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(body.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch")
    results = [predict(t) for t in body.texts]
    return {"results": results, "count": len(results)}
