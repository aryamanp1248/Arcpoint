# Arcpoint Context Layer Prototype

## Overview

This project implements a prototype **Context Layer** for intelligent routing of LLM inference requests. The Context Layer acts as a real-time intelligence service that aggregates system signals, reasons over historical and current state, and answers operational questions needed by a routing engine or LLM-based agent.

The design prioritizes:

* **Time-aware reasoning** over model performance, traffic, and quality
* **Deterministic filtering and scoring** before involving an LLM
* **Fast, unified context access** via a single API endpoint
* **Explainability**, with evidence-backed answers instead of opaque decisions

Rather than functioning as a CRUD service or metrics dashboard, the Context Layer is treated as a **queryable control-plane primitive** that supports routing decisions, investigations, and capacity awareness.

---

## Approach

### Unified Context Interface

The system exposes a single API endpoint that accepts natural-language queries. These queries may relate to:

* model or backend suitability for a request
* historical quality regressions
* near-term traffic patterns
* fleet availability and constraints

This makes the Context Layer easy to consume by:

* a routing engine
* an LLM-based agent
* or a human operator

All intelligence lives in the Context Engine; the API layer is intentionally thin.

---

### Deterministic First, LLM Second

The system uses **deterministic logic** wherever correctness matters:

* model viability checks (status, SLA, staleness, rate limits)
* heuristic scoring
* stratified Top-N candidate selection
* historical aggregation and comparison

LLMs are used only for:

* classifying query intent
* generating natural-language explanations from structured evidence

Strict prompt guardrails ensure that responses do not speculate or hallucinate beyond the available data.

---

### Time as a First-Class Dimension

All analysis is anchored to an internally computed **system time**, derived from observed request logs. This allows the system to consistently answer questions such as:

* “two days ago”
* “last hour”
* “recent trend vs baseline”

Historical quality analysis compares windows against prior periods rather than relying on absolute thresholds.

---

### Synthetic but Realistic Data

The project includes a data generator that produces multi-day request logs with:

* multiple user tiers and SLAs
* diverse task types
* latency and failure modeling
* intentional quality degradation
* traffic variability

This enables meaningful demonstrations of routing trade-offs, quality drift, and forecasting behavior without relying on live infrastructure.

---

## How to Run the Project

### 1. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

---

### 2. Generate Synthetic Request Data

This creates a realistic multi-day request log used by the Context Engine:

```bash
python generate_requests.py
```

The output is written to:

```
app/data/mock_requests.csv
```

---

### 3. Start the Context API

Run the FastAPI service:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```
http://localhost:8000
```

---

### 4. Query the Context Layer

Example query:

```bash
curl -X POST http://localhost:8000/v1/context/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "operator_1",
    "query": "Why did quality drop for reasoning tasks two days ago?"
  }'
```

The response will be a natural-language answer grounded strictly in system evidence.

---

## Notes

* This prototype uses **offline synthetic data** to demonstrate behavior.
* There is no persistent state store or streaming ingestion.
* The system is designed to be extended with real telemetry, caching layers, and decision tracing.

---

## Commit History

This repository reflects a completed prototype submitted as a single commit due to time constraints. Design intent, trade-offs, and execution details are conveyed through the implementation and documentation.

---

