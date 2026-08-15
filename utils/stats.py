import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
TRACE_FILE = Path(os.getenv("RAG_TRACE_FILE", BASE_DIR / "evaluation" / "traces.jsonl"))
EVAL_SUMMARY_FILE = BASE_DIR / "evaluation" / "summary.json"


def trace_file_fingerprint() -> Tuple[float, int]:
    """(mtime, size) of the trace log, for cache invalidation by callers."""
    if not TRACE_FILE.exists():
        return (0.0, 0)
    file_stat = TRACE_FILE.stat()
    return (file_stat.st_mtime, file_stat.st_size)


def read_trace_events() -> List[Dict[str, Any]]:
    """Parse evaluation/traces.jsonl line by line, skipping malformed lines."""
    if not TRACE_FILE.exists():
        return []

    events = []
    with TRACE_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def aggregate_usage_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll per-query trace events up into KPIs + a daily series for charts."""
    total_queries = 0
    total_errors = 0
    latencies: List[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    cost_known = False
    fallback_count = 0
    diagnostics_count = 0
    rerank_scores: List[float] = []
    daily_counts: Dict[str, int] = {}
    daily_cost: Dict[str, float] = {}

    for event in events:
        step = event.get("step")
        outputs = event.get("outputs") or {}
        timestamp = event.get("timestamp")
        day = timestamp[:10] if isinstance(timestamp, str) and len(timestamp) >= 10 else None

        if step == "trace_end":
            total_queries += 1
            if outputs.get("status") == "error":
                total_errors += 1
            latency = outputs.get("latency_seconds")
            if isinstance(latency, (int, float)):
                latencies.append(latency)
            if day:
                daily_counts[day] = daily_counts.get(day, 0) + 1

        elif step == "llm_response":
            token_usage = outputs.get("token_usage") or {}
            total_input_tokens += token_usage.get("input_tokens") or 0
            total_output_tokens += token_usage.get("output_tokens") or 0
            cost = outputs.get("estimated_cost_usd")
            if isinstance(cost, (int, float)):
                total_cost += cost
                cost_known = True
                if day:
                    daily_cost[day] = daily_cost.get(day, 0.0) + cost

        elif step == "retrieval_diagnostics":
            diagnostics_count += 1
            if outputs.get("weakness_reason"):
                fallback_count += 1
            for chunk in outputs.get("reranked_chunks") or []:
                score = chunk.get("rerank_score")
                if isinstance(score, (int, float)):
                    rerank_scores.append(score)

    all_days = sorted(set(daily_counts) | set(daily_cost))
    daily_series = [
        {
            "date": day,
            "queries": daily_counts.get(day, 0),
            "cost_usd": round(daily_cost.get(day, 0.0), 6),
        }
        for day in all_days
    ]

    return {
        "total_queries": total_queries,
        "total_errors": total_errors,
        "avg_latency_seconds": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4) if cost_known else None,
        "fallback_rate": (
            round(fallback_count / diagnostics_count, 4) if diagnostics_count else None
        ),
        "avg_rerank_score": (
            round(sum(rerank_scores) / len(rerank_scores), 3) if rerank_scores else None
        ),
        "daily_series": daily_series,
    }


def read_eval_summary() -> Optional[Dict[str, Any]]:
    """Last `python -m evaluation.run_eval` snapshot, or None if never run."""
    if not EVAL_SUMMARY_FILE.exists():
        return None
    try:
        with EVAL_SUMMARY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    data["_generated_at"] = datetime.fromtimestamp(
        EVAL_SUMMARY_FILE.stat().st_mtime
    ).isoformat()
    return data
