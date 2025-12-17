import pandas as pd
import random
import uuid
import json
from datetime import datetime, timedelta

# Load snapshot-based model metadata
with open("app/data/model_state.json", "r") as f:
    model_state_snapshot = json.load(f)

# Extract models list from snapshot
model_states = model_state_snapshot.get("models", [])

# Extract model info
model_ids = [model["model_id"] for model in model_states]
model_info = {model["model_id"]: model for model in model_states}

# User tiers and SLA (ms)
user_tiers = {
    "free": 800,
    "pro": 300,
    "enterprise": 150
}

# Supported task types
task_types = ["reasoning", "summarization", "classification", "code-gen", "qa"]

# Simulate request time across 5 days
start_time = datetime.now() - timedelta(days=5)
rows = []

for _ in range(10000):
    user_id = f"user_{random.randint(1, 500)}"
    tier = random.choices(list(user_tiers.keys()), weights=[0.6, 0.3, 0.1])[0]
    sla = user_tiers[tier]
    task_type = random.choice(task_types)

    model = random.choice(model_states)
    model_id = model["model_id"]
    backend = model["backend"]
    max_rps = model["max_throughput_rps"]
    spot_available = model["spot_instance_available"]
    quality_trend = model["quality_score_trend"]

    # Simulate timestamp
    timestamp = start_time + timedelta(seconds=random.randint(0, 5 * 24 * 3600))

    # Forecasted quality drop: apply 15% penalty on day 4
    days_since = (timestamp - start_time).days
    degradation_penalty = 0.0
    if days_since == 4 and random.random() < 0.4:
        degradation_penalty = 0.15

    # Base quality score
    base_quality = random.uniform(0.75, 0.97)

    # Apply quality trend using recent valid signal history
    trend_mod = 0.0
    if quality_trend and isinstance(quality_trend, list):
        recent_trend = quality_trend[-3:]
        cleaned = [x for x in recent_trend if isinstance(x, (int, float))]
        if cleaned:
            trend_mod = sum(cleaned) / len(cleaned)

    quality_score = max(
        0.0,
        min(1.0, base_quality + trend_mod - degradation_penalty)
    )

    # Latency modeling based on SLA
    latency = int(random.gauss(mu=sla * 0.7, sigma=50))
    latency = max(30, min(latency, int(sla * 2)))

    # Simulate backpressure, SLA breach, and failure
    success_chance = 0.98
    if latency > sla * 1.5 or degradation_penalty > 0 or random.random() < 0.03:
        success_chance -= 0.2

    success = random.random() < success_chance

    rows.append({
        "request_id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "user_id": user_id,
        "tier": tier,
        "sla_ms": sla,
        "task_type": task_type,
        "model_id": model_id,
        "backend": backend,
        "latency_ms": latency,
        "quality_score": round(quality_score, 3),
        "success": success
    })

df = pd.DataFrame(rows)
df.to_csv("app/data/mock_requests.csv", index=False)

print("✅ 10,000 realistic mock requests generated at app/data/mock_requests.csv")
