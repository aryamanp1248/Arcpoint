# Arcpoint Context Layer Prototype

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

## Repository Structure

```
Arcpoint/
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

## Setup

### 1. Create and Activate a Python Virtual Environment

From the project root:

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

You should see `(.venv)` in your terminal.

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure OpenAI API Key

This project uses the OpenAI API for:

* query intent classification
* evidence-backed natural language responses

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key_here
ROUTER_LLM_MODEL=gpt-4o-mini
```

Notes:

* `ROUTER_LLM_MODEL` is optional and defaults to `gpt-4o-mini`

---

## Generate Synthetic Data

The Context Engine reads request history from:

```
app/data/mock_requests.csv
```

Generate it using:

```bash
python generate_requests.py
```

This creates **10,000 realistic requests** across ~5 days with:

* multiple user tiers and SLAs
* diverse task types
* latency and failure modeling
* intentional quality degradation for investigation queries

---

## Run the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The service will be available at:

```
http://localhost:8000
```

---

## Using Swagger / OpenAPI UI

FastAPI automatically exposes an interactive Swagger UI.

### Open Swagger UI

Once the server is running, open your browser and go to:

```
http://localhost:8000/docs
```

---

### How to Submit a Query via Swagger

1. Expand the endpoint:

   ```
   POST /v1/context/query
   ```

2. Click **“Try it out”**

3. Enter request payload in the body field:

```json
{
  "user_id": "operator_1",
  "query": "Why did quality drop for reasoning tasks two days ago?"
}
```

4. Click **Execute**

5. View:

   * the generated response
   * returned natural-language explanation
   * HTTP status and latency

Swagger UI is the recommended way to explore the Context Layer interactively without using curl.

---

## Querying via cURL (Optional)

```bash
curl -X POST http://localhost:8000/v1/context/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "operator_1",
    "query": "What will traffic look like in the next hour?"
  }'
```

---

## How It Works (High-Level)

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


