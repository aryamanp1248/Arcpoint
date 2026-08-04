# ModelPilot Context Layer Prototype

## Overview

This repository contains a working prototype of a **Context Layer** for intelligent routing of LLM inference requests. The Context Layer aggregates system signals (model snapshot + request history), maintains a time-aware view of state, and answers operational questions needed by a routing engine or LLM-based agent.

The prototype supports natural-language questions such as:

* Which models are viable for this task and SLA?
* Why did quality drop for a task on a given day?
* What will traffic look like in the next hour?

The design emphasizes:

* **Deterministic filtering and scoring** before any LLM involvement
* **Time-aware analysis** using historical request logs
* **Unified agent-friendly API** via a single endpoint
* **Evidence-backed explanations** (guardrails prevent hallucination)

---

## Live Demo

**Live Demo:** https://modelpilot-chix.onrender.com

**Interactive API Docs (Swagger UI):** https://modelpilot-chix.onrender.com/docs

> **Note:** The app is hosted on Render's free tier and may take ~30 seconds to wake up after a period of inactivity.

### Example Queries To Try

| Query Type | Example |
|---|---|
| Routing | "Which model should I use for a code-gen task with a 300ms SLA?" |
| Quality Investigation | "Why did quality drop for reasoning tasks two days ago?" |
| Forecasting | "What will traffic look like in the next hour?" |
| Fleet Status | "What models are available for summarization?" |

---

## Repository Structure

```
ModelPilot/
├── app/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── api/
│   │   └── routes.py              # POST /v1/context/query endpoint
│   ├── services/
│   │   └── context_engine.py       # Core logic: routing, quality analysis, forecasting
│   └── data/
│       ├── model_state.json        # Snapshot of model fleet state/metadata
│       └── mock_requests.csv       # Generated request log
├── generate_requests.py            # Generates synthetic request traffic
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Prevents secrets / venv from being committed
└── README.md
```

---

## How It Works

* `ContextEngine` loads:

  * model snapshot (`model_state.json`)
  * historical request log (`mock_requests.csv`)
* An LLM classifies query intent into:

  * `route`, `forecast`, `quality_issue`, or `models`
* Deterministic logic assembles evidence:

  * viability checks + stratified Top-N routing
  * historical quality comparison vs baseline
  * traffic forecast using rolling windows
* The LLM produces a natural-language answer using **only the provided evidence**

---

## Local Setup

> The API is already deployed and live. The instructions below are only needed if you want to run the project locally.

### 1. Create and Activate a Python Virtual Environment

#### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (CMD)

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
ROUTER_LLM_MODEL=gpt-4o-mini
```

> `ROUTER_LLM_MODEL` is optional and defaults to `gpt-4o-mini`

---

### 4. Generate Synthetic Data

```bash
python generate_requests.py
```

This creates **10,000 realistic requests** across ~5 days with:

* multiple user tiers and SLAs
* diverse task types
* latency and failure modeling
* intentional quality degradation for investigation queries

---

### 5. Run the API Locally

```bash
uvicorn app.main:app --reload
```

The service will be available at `http://localhost:8000` and Swagger UI at `http://localhost:8000/docs`

---

## Querying via cURL (Optional)

**Against the live deployment:**

```bash
curl -X POST https://modelpilot-chix.onrender.com/v1/context/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "operator_1",
    "query": "What will traffic look like in the next hour?"
  }'
```

**Against a local instance:**

```bash
curl -X POST http://localhost:8000/v1/context/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "operator_1",
    "query": "What will traffic look like in the next hour?"
  }'
```

---

## Deployment

Deployed on **Render** as a live web service.

* **Platform:** Render (Free Tier)
* **Live URL:** https://modelpilot-chix.onrender.com
* **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
* **Environment Variables:** `OPENAI_API_KEY` configured via Render dashboard
