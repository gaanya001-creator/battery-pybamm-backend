# PyBaMM Battery Backend

FastAPI + PyBaMM backend for Battery Simulator v11/v12.

## Deploy on Render (Free)

1. GitHub pe push karo (server.py, requirements.txt, render.yaml)
2. [render.com](https://render.com) → New → Web Service
3. GitHub repo connect karo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Deploy → URL copy karo
6. Simulator mein Backend URL field mein paste karo → PING

## Local Run

```bash
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

## Test

```
GET  /health    → server status
POST /simulate  → run PyBaMM simulation
```

## Bug Fixes in v2.0
- initial_soc default: 0.5 → 1.0 (full charge)
- V_mid returned for accurate comparison (not just V_end cutoff)
- cap_mAh sanity bound: 10000 → 50000
- Proper try/except with DFN → SPMe auto-fallback
- Peukert exponent now chemistry-specific
