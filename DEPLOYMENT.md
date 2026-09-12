# Vercel Deployment Guide (GaTi)

## 1) Deploy prerequisites
- Python dependencies from `requirements.txt`
- Frontend static build from `frontend/package.json`

## 2) Environment variables
Set these in Vercel Project Settings → Environment Variables:
- `RAILRADAR_API_KEY`
- `RAILRADAR_BASE_URL` (default: `https://api.railradar.in/v1`)
- `LOG_LEVEL` (optional, default `INFO`)
- `PREDICTION_LOG_PATH` (optional; use `/tmp/gati_prediction_eval_log.jsonl` on Vercel)

## 3) How this repository is wired for Vercel
- `vercel.json` routes all `/api/*` requests to `api/index.py` (FastAPI serverless function)
- `frontend/package.json` builds static dashboard files into `frontend/dist`
- `src/api/main.py` includes:
  - `GET /api/health` for platform health checks
  - structured logging for serverless runtime
  - safe default log path fallback to `/tmp` on Vercel

## 4) Local development
Use Docker:
```bash
docker compose up --build
```
API health check:
```bash
curl http://localhost:8000/api/health
```

## 5) Notes for serverless runtime
- The prediction model and processed data are bundled from repository files (`models/`, `data/`)
- Runtime writes should use `/tmp` on Vercel
- Cold starts are reduced by module-level FastAPI/simulator initialization in `src/api/main.py`
