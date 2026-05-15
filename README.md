# MentalRoBERTa FastAPI — Railway Deployment Guide

## Project Structure

```
your-project/
├── main.py
├── requirements.txt
├── Dockerfile
├── railway.toml
├── .dockerignore
├── .gitignore
└── mental_roberta_final/      ← your fine-tuned model folder
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    ├── model.safetensors (or pytorch_model.bin)
    └── special_tokens_map.json
```

---

## Step-by-Step Deploy to Railway (Free Tier)

### 1. Push code to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

> ⚠️ **Important:** Make sure `mental_roberta_final/` folder is included in the commit.
> The `.gitignore` is set to NOT ignore it.
> GitHub has a 100MB file size limit — if your model file is larger, see Option B below.

### 2. Create a Railway project

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project → Deploy from GitHub repo**
3. Select your repository
4. Railway will auto-detect the `Dockerfile` and start building

### 3. Set environment variables (optional)

In Railway dashboard → your service → **Variables**, you can set:

| Variable | Value | Notes |
|----------|-------|-------|
| `MODEL_PATH` | `./mental_roberta_final` | Default, points to bundled model |
| `PORT` | (leave empty) | Railway sets this automatically |

### 4. Wait for build & deploy

- First build takes **5–10 minutes** (downloading PyTorch + model)
- After that, Railway gives you a public URL like:
  `https://your-app.up.railway.app`

---

## API Usage

### Single prediction
```bash
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been feeling very anxious lately"}'
```

### Batch prediction
```bash
curl -X POST https://your-app.up.railway.app/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I feel hopeless", "I am doing great today"]}'
```

### Health check
```bash
curl https://your-app.up.railway.app/health
```

---

## Option B — Model too large for GitHub (>100MB)?

If your `pytorch_model.bin` or `model.safetensors` is over 100MB:

### Use Git LFS
```bash
git lfs install
git lfs track "*.bin" "*.safetensors"
git add .gitattributes
git add mental_roberta_final/
git commit -m "add model with LFS"
git push
```

### OR — Load from HuggingFace Hub instead of bundling

If you upload your fine-tuned model to HuggingFace Hub:
1. Push your model: `model.push_to_hub("your-username/mental-roberta-finetuned")`
2. In Railway Variables, set:
   - `MODEL_PATH` = `your-username/mental-roberta-finetuned`
3. Remove the `COPY mental_roberta_final/` line from `Dockerfile`

---

## Railway Free Tier Limits

| Resource | Limit |
|----------|-------|
| RAM | 512MB (Hobby: 8GB) |
| vCPU | Shared |
| Sleep | After inactivity (free tier sleeps) |
| Egress | 100GB/month |

> ⚠️ The base `mental-roberta-base` model is ~500MB in memory.
> Free tier (512MB RAM) may be tight — consider upgrading to Hobby ($5/month) for reliability.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails on torch | Make sure requirements.txt has CPU-only torch (no `+cu118`) |
| OOM crash | Upgrade to Railway Hobby tier for 8GB RAM |
| `/health` returns 502 | Model still loading — wait 2–3 min after deploy |
| Model not found error | Check `mental_roberta_final/` folder is committed to git |
