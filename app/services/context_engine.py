
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
from openai import OpenAI
# =============================
# Config
# =============================
MODEL_STATE_PATH = "app/data/model_state.json"
REQUEST_LOG_PATH = "app/data/mock_requests.csv"
DEFAULT_TOP_N = 5
DEFAULT_LLM_MODEL = os.getenv("ROUTER_LLM_MODEL", "gpt-4o-mini")
MAX_STALENESS_MINUTES = 24 * 60 * 7
LATENCY_HARD_MULTIPLIER = 2.0
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ContextEngine:
    def __init__(self):
        self.model_state: Dict[str, Any] = {}
        self.request_log_df: pd.DataFrame = pd.DataFrame()
        self.load_model_state()
        self.load_request_log()
    # =============================
    # Loading
    # =============================
    def load_model_state(self):
        with open(MODEL_STATE_PATH, "r") as f:
            self.model_state = json.load(f)
    def load_request_log(self):
        df = pd.read_csv(REQUEST_LOG_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self.request_log_df = df
    # =============================
    # Time anchor
    # =============================
    def _system_now(self) -> datetime:
        if self.request_log_df.empty:
            return datetime.now(timezone.utc)
        return self.request_log_df["timestamp"].max().to_pydatetime()
    # =============================
    # ---------- Helpers ----------
    # =============================
    def _staleness_minutes(self, iso_ts: Optional[str]) -> Optional[float]:
        if not iso_ts:
            return None
        return (self._system_now() - pd.to_datetime(iso_ts, utc=True)).total_seconds() / 60
    def _clean_trend_sum(self, trend: Any) -> float:
        if not isinstance(trend, list):
            return 0.0
        vals = [x for x in trend if isinstance(x, (int, float))]
        return float(sum(vals)) if vals else 0.0
    def _recent_success_rate(self, model_id: str, hours: int = 24) -> Optional[float]:
        now = self._system_now()
        df = self.request_log_df[
            (self.request_log_df["timestamp"] >= now - timedelta(hours=hours))
            & (self.request_log_df["model_id"] == model_id)
        ]
        return None if df.empty else float(df["success"].mean())
    def _recent_median_quality(self, model_id: str, task_type: str, hours: int = 24) -> Optional[float]:
        now = self._system_now()
        df = self.request_log_df[
            (self.request_log_df["timestamp"] >= now - timedelta(hours=hours))
            & (self.request_log_df["model_id"] == model_id)
            & (self.request_log_df["task_type"] == task_type)
        ]
        return None if df.empty else float(df["quality_score"].median())
    # =============================
    # ---------- Intent ----------
    # =============================
    def classify_operator_query(self, query: str) -> Dict[str, Any]:
        today = self._system_now().strftime("%Y-%m-%d")
        prompt = f"""
TODAY = {today}
Classify the user query into exactly ONE action:
- route → asking which model to use 
- forecast → future load related questions
- quality_issue → past performance related questions
- models → asking about available models OR backends to use for a particular task
Extract parameters when possible.
Return ONLY JSON:
{{
  "action": "route|forecast|quality_issue|models",
  "params": {{
    "days_ago": <int|null>,
    "hours": <int|null>,
    "task_type": <string|null>,
    "sla_ms": <int|null>,
    "needs_long_context": <bool>,
    "priority": "quality|latency|cost|reliability|balanced"
  }},
  "confidence": "high|medium|low"
}}
User query:
"{query}"
"""
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    # =============================
    # ---------- Viability ----------
    # =============================
    def _is_viable(self, m: Dict[str, Any], sla_ms: int) -> bool:
        if m.get("status") == "down":
            return False
        lat = m.get("latency_ms")
        if lat is None or lat > sla_ms * LATENCY_HARD_MULTIPLIER:
            return False
        stale = self._staleness_minutes(m.get("last_updated"))
        if stale and stale > MAX_STALENESS_MINUTES:
            return False
        rl = m.get("rate_limit_remaining")
        if isinstance(rl, (int, float)) and rl <= 0:
            return False
        return True
    # =============================
    # ---------- Capability Tags ----------
    # =============================
    def _capability_tags(self, m: Dict[str, Any], sla_ms: int) -> List[str]:
        tags = list(m.get("tasks_supported", []))
        if m.get("context_window", 0) >= 32000:
            tags.append("long_context")
        if m.get("latency_ms", sla_ms * 2) <= min(200, sla_ms * 0.8):
            tags.append("low_latency")
        return list(dict.fromkeys(tags))
    # =============================
    # ---------- Heuristic ----------
    # =============================
    def _heuristic_score(self, m: Dict[str, Any], task: str, sla_ms: int) -> float:
        lat = m.get("latency_ms", sla_ms * 2) / sla_ms
        cost = m.get("cost_per_1k", 10) * 10
        err = (m.get("error_rate") or 0.05) * 3
        status_penalty = 0.25 if m.get("status") == "degraded" else 0
        trend_penalty = -self._clean_trend_sum(m.get("quality_score_trend"))
        return lat + cost + err + status_penalty + trend_penalty
    # =============================
    # ---------- Stratified Top-N ----------
    # =============================
    def get_candidate_models(
        self,
        task_type: str,
        sla_ms: int,
        n: int = DEFAULT_TOP_N,
        needs_long_context: bool = False,
        priority: str = "balanced",
    ) -> List[Dict[str, Any]]:
        models = self.model_state.get("models", [])
        viable = [m for m in models if self._is_viable(m, sla_ms)]
        if not viable:
            return []
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for m in viable:
            for tag in self._capability_tags(m, sla_ms):
                buckets.setdefault(tag, []).append(m)
        quotas = {task_type: 2}
        if needs_long_context:
            quotas["long_context"] = 1
        quotas.setdefault("qa", 1)
        quotas.setdefault("chat", 1)
        picked, seen = [], set()
        for cap, k in quotas.items():
            ranked = sorted(
                buckets.get(cap, []),
                key=lambda m: self._heuristic_score(m, task_type, sla_ms)
            )
            for m in ranked:
                if m["model_id"] in seen:
                    continue
                if task_type not in m.get("tasks_supported", []):
                    continue
                picked.append(self._enrich_candidate(m, task_type, sla_ms, f"bucket:{cap}"))
                seen.add(m["model_id"])
                if len(picked) >= n:
                    return picked
        return picked[:n]
    def _enrich_candidate(self, m: Dict[str, Any], task: str, sla_ms: int, why: str) -> Dict[str, Any]:
        return {
            **m,
            "why": why,
            "heuristic_score": round(self._heuristic_score(m, task, sla_ms), 4),
            "capability_tags": self._capability_tags(m, sla_ms),
        }
    # =============================
    # ---------- Quality Dip ----------
    # =============================
    def resolve_day_window(self, days_ago: int) -> Dict[str, str]:
        now = self._system_now().date()
        target = now - timedelta(days=days_ago)

        start = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        return {
            "start_ts": start.isoformat().replace("+00:00", "Z"),
            "end_ts": end.isoformat().replace("+00:00", "Z"),
        }

    # =============================
    # Quality dip analysis (comparative)
    # =============================
    def explain_quality_dip(self, start_ts: str, end_ts: str) -> Dict[str, Any]:
        start = pd.to_datetime(start_ts, utc=True)
        end = pd.to_datetime(end_ts, utc=True)

        window = self.request_log_df[
            (self.request_log_df["timestamp"] >= start)
            & (self.request_log_df["timestamp"] < end)
        ]
        if window.empty:
            return {
                "status": "no_data",
                "start_ts": start_ts,
                "end_ts": end_ts,
                "message": "No requests found in the specified time window.",
            }

        baseline_start = start - timedelta(days=1)
        baseline_end = end - timedelta(days=1)
        baseline = self.request_log_df[
            (self.request_log_df["timestamp"] >= baseline_start)
            & (self.request_log_df["timestamp"] < baseline_end)
        ]

        grp = ["model_id", "task_type"]
        w = (
            window.groupby(grp)
            .agg(
                avg_quality=("quality_score", "mean"),
                success_rate=("success", "mean"),
                volume=("request_id", "count"),
                avg_latency=("latency_ms", "mean"),
            )
            .reset_index()
        )

        if baseline.empty:
            w["quality_delta_vs_prev_day"] = None
            ranked = w.sort_values(["avg_quality", "volume"], ascending=[True, False]).head(10)
        else:
            b = (
                baseline.groupby(grp)
                .agg(base_quality=("quality_score", "mean"))
                .reset_index()
            )
            m = w.merge(b, on=grp, how="left")
            m["quality_delta_vs_prev_day"] = m["avg_quality"] - m["base_quality"]
            m["volume_gate"] = m["volume"].apply(lambda x: x >= 15)
            ranked = m.sort_values(["volume_gate", "quality_delta_vs_prev_day"], ascending=[False, True]).head(10)

        state_by_id = {x["model_id"]: x for x in (self.model_state.get("models", []) or [])}
        offenders: List[Dict[str, Any]] = []
        for _, r in ranked.iterrows():
            mid = r["model_id"]
            ms = state_by_id.get(mid, {})
            offenders.append({
                "model_id": mid,
                "task_type": r["task_type"],
                "avg_quality": float(r["avg_quality"]),
                "success_rate": float(r["success_rate"]),
                "volume": int(r["volume"]),
                "avg_latency_ms": float(r["avg_latency"]),
                "quality_delta_vs_prev_day": None if pd.isna(r.get("quality_delta_vs_prev_day")) else float(r.get("quality_delta_vs_prev_day")),
                "model_status": ms.get("status"),
                "backend": ms.get("backend"),
                "error_rate": ms.get("error_rate"),
                "maintenance_window": ms.get("maintenance_window"),
                "quality_trend_sum": self._clean_trend_sum(ms.get("quality_score_trend")),
            })

        return {
            "status": "ok",
            "start_ts": start_ts,
            "end_ts": end_ts,
            "offenders": offenders,
        }

    # =============================
    # ---------- Forecast ----------
    # =============================
    def forecast_traffic(self, hours: int = 1) -> Dict[str, Any]:
            if self.request_log_df.empty:
                return {"error": "Insufficient data", "confidence": "low"}

            df = self.request_log_df.set_index("timestamp")
            hourly = df.resample("1H").size()

            recent = hourly.tail(12)  # 12h window
            mean = float(recent.mean())
            std = float(recent.std())
            cv = (std / mean) if mean > 0 else 0.0

            confidence = "high" if cv < 0.15 else "medium" if cv < 0.35 else "low"
            last = float(recent.iloc[-1]) if len(recent) > 0 else mean
            spike = (last > mean * 1.4) if mean > 0 else False

            return {
                "predicted_requests": int(round(mean * hours)),
                "hours": int(hours),
                "avg_requests_per_hour": int(round(mean)),
                "volatility_cv": round(cv, 2),
                "spike_detected_recent_hour": bool(spike),
                "confidence": confidence,
                "method": "rolling_mean_12h",
            }
    
    # =============================
    # ---------- Natural Language ----------
    # =============================
    def _nl_answer(self, query: str, evidence: Dict[str, Any]) -> str:
        prompt = f"""
You are an assistant for an AI routing control plane. 
CRITICAL RULES: 
- ONLY use facts explicitly present in the Evidence JSON. 
- Do NOT mention models, incidents, or statuses that do not appear in Evidence. 
- Do NOT speculate or infer causes beyond what Evidence supports. 
- Do NOT include exact numeric deltas. Describe trends qualitatively (e.g., ‘largest decline’, ‘moderate drop’) but mention which attribute is affected.
- If the question refers to a past time window, do NOT use model_state metadata such as quality_score_trend, status, or maintenance windows.
- If question has "_ days ago" in it, EXPLICITLY mention the date in DD-MM-YYYY
User question:
"{query}"
Evidence:
{json.dumps(evidence, indent=2)}
Write a natural language answer:
- Answer questions directly first
- Explain briefly using ONLY the evidence
- If evidence is insufficient, then say so
"""
        r = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    # =============================
    # ---------- Public API ----------
    # =============================
    def answer(self, query: str) -> str:
        intent = self.classify_operator_query(query)
        action = intent["action"]
        params = intent.get("params", {}) or {}

        if action == "forecast":
            ev = self.forecast_traffic(params.get("hours", 1))
            return self._nl_answer(query, ev)

        if action == "quality_issue":
            days_ago = params.get("days_ago")
            days_ago = 2 if days_ago is None else int(days_ago)
            window = self.resolve_day_window(days_ago)
            dip = self.explain_quality_dip(window["start_ts"], window["end_ts"])
            evidence = {
                "action": "quality_issue",
                "system_now": self._system_now().isoformat(),
                "window": window,
                "analysis": dip,
            }
            return self._nl_answer(query, evidence)

        if action == "models":
            return self._nl_answer(query, self.model_state)

        # Default: routing
        task = params.get("task_type")
        if not task:
            task = "reasoning"
        sla = params.get("sla_ms")
        if not sla:
            sla = 300
        cands = self.get_candidate_models(task, sla)
        if not cands:
            return "No viable models match the task and SLA."
        return self._nl_answer(query, {"candidates": cands})
    