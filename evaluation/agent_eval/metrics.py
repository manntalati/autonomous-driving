import re
from typing import Any

from agent.loop import AgentResult
from evaluation.agent_eval.benchmark import BenchQuestion

_PRICE_IN  = 15.0 / 1_000_000
_PRICE_OUT = 75.0 / 1_000_000

def parse_count(answer: str) -> int | None:
    match = re.search(r'\d+', answer)
    if not match:
        return None
    return int(match.group())

def parse_distance(answer: str) -> float | None:
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b', answer, re.IGNORECASE)
    if m: return float(m.group(1))
    else: return None

def parse_yesno(answer: str) -> bool | None:
    a = answer.strip().lower()
    if a.startswith("yes"): return True
    if a.startswith("no"):  return False
    return None

def score_answer(q: BenchQuestion, r: AgentResult) -> dict:
    pred: Any = None
    if q.qtype == "count":
        pred = parse_count(r.answer)
        correct = (pred is not None) and abs(pred - q.gt_answer) <= 1
    elif q.qtype == "presence":
        pred = parse_yesno(r.answer)
        correct = (pred is not None) and (pred == q.gt_answer)
    elif q.qtype == "nearest":
        pred = parse_distance(r.answer)
        if pred is None or q.gt_answer == 0:
            correct = False
        else:
            correct = abs(pred - q.gt_answer) / q.gt_answer <= 0.20
    elif q.qtype == "spatial":
        pred = parse_yesno(r.answer)
        correct = (pred is not None) and (pred == q.gt_answer)
    else:
        raise ValueError(f"unknown qtype: {q.qtype}")
    return {
        "correct": bool(correct),
        "parse_failed": pred is None,
        "pred": pred,
        "gt": q.gt_answer,
    }
    

def score_tool_selection(q: BenchQuestion, r: AgentResult) -> dict:
    called = set(r.trace)
    expected = set(q.expects_tools)
    intersect = called & expected
    precision = len(intersect) / len(called) if called else 1.0
    recall = len(intersect) / len(expected) if expected else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "exact_match": called == expected,
        "n_calls": len(r.trace)
    }

def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}

    def _mean(key): return sum(r[key] for r in results) / n
    def _pct(key, p):
        xs = sorted(r[key] for r in results)
        k = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
        return xs[k]

    total_in  = sum(r["input_tokens"]  for r in results)
    total_out = sum(r["output_tokens"] for r in results)

    return {
        "n": n,
        "accuracy":        _mean("correct"),
        "parse_failures":  sum(r["parse_failed"] for r in results) / n,
        "mean_tool_calls": _mean("n_calls"),
        "tool_precision": _mean("precision"),
        "tool_recall":    _mean("recall"),
        "p50_latency_s":  _pct("latency_s", 50),
        "p95_latency_s":  _pct("latency_s", 95),
        "total_input_tokens":  total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": total_in * _PRICE_IN + total_out * _PRICE_OUT,
        "cost_per_query_usd": (total_in * _PRICE_IN + total_out * _PRICE_OUT) / n,
    }
    