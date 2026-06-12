from evaluation.agent_eval.benchmark import build_benchmark, BenchQuestion
from evaluation.agent_eval.metrics import aggregate
from agent.mcp_client import MCPClient
from agent.loop import run_agent, AgentResult
from evaluation.agent_eval.metrics import score_answer, score_tool_selection
import time
import asyncio
import json
import argparse
from pathlib import Path
from agent.config import load_env
from agent.prompts import SYSTEM_PROMPT
from data.dataset import MINI_VAL_SCENES
from nuscenes.nuscenes import NuScenes

async def run_one(q: BenchQuestion, client: MCPClient, model: str, system: str | None) -> dict:
    t0 = time.perf_counter()
    try:
        result = await run_agent(q.question, client, model=model, system=system)
    except Exception as e:
        return {
            "qid": q.qid, "qtype": q.qtype, "error": str(e),
            "answer": "", "trace": [], "turns": 0,
            "input_tokens": 0, "output_tokens": 0,
            "latency_s": time.perf_counter() - t0,
            "correct": False, "parse_failed": True, "pred": None, "gt": q.gt_answer,
            "precision": 0.0, "recall": 0.0, "exact_match": False, "n_calls": 0,
        }
    latency = time.perf_counter() - t0
    return {
        "qid": q.qid, "qtype": q.qtype,
        "answer": result.answer, "trace": result.trace, "turns": result.turns,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "latency_s": latency,
        **score_answer(q, result),
        **score_tool_selection(q, result),
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of questions (for smoke tests)")
    parser.add_argument("--no-system", action="store_true",
                        help="ablation: disable the system prompt")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="max parallel agent calls; caps Anthropic rate")
    parser.add_argument("--data-root", default="data/raw/v1.0-mini")
    parser.add_argument("--results-path", default="logs/agent_eval_results.jsonl")
    parser.add_argument("--report-path",  default="logs/agent_eval_run.log")
    args = parser.parse_args()

    load_env()

    # 1. build benchmark (or load cache if present — saves a few seconds on reruns)
    bench_path = Path("logs/agent_eval_benchmark.json")
    if bench_path.exists():
        questions = [BenchQuestion(**q) for q in json.loads(bench_path.read_text())]
    else:
        nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)
        questions = build_benchmark(nusc)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Running {len(questions)} questions ({args.model}, "
          f"system={'off' if args.no_system else 'on'})")

    # 2. connect ONE MCP session — SceneStore caches frame loads across calls.
    client = MCPClient()
    await client.connect()
    system = None if args.no_system else SYSTEM_PROMPT

    # 3. semaphore-bounded concurrency (don't slam Anthropic rate limits)
    sem = asyncio.Semaphore(args.concurrency)
    async def _bounded(q):
        async with sem:
            return await run_one(q, client, args.model, system)

    try:
        results = await asyncio.gather(*[_bounded(q) for q in questions])
    finally:
        await client.close()

    # 4. per-question results → JSONL (one dict per line; easy to grep later)
    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # 5. aggregate + write the report
    summary = aggregate(results)
    _write_report(args.report_path, summary, args)
    print(f"Wrote {args.results_path} and {args.report_path}")
    print(f"Accuracy {summary['accuracy']:.3f}   "
          f"tool calls {summary['mean_tool_calls']:.1f}   "
          f"$/q {summary['cost_per_query_usd']:.4f}")


def _write_report(path: str, summary: dict, args) -> None:
    """Markdown table in the same style as the per-phase training logs."""
    lines = [
        f"# Agent Eval — model={args.model}, system={'off' if args.no_system else 'on'}",
        "",
        f"N questions: {summary['n']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| accuracy | {summary['accuracy']:.3f} |",
        f"| parse failures | {summary['parse_failures']:.3f} |",
        f"| mean tool calls | {summary['mean_tool_calls']:.2f} |",
        f"| tool precision | {summary['tool_precision']:.3f} |",
        f"| tool recall | {summary['tool_recall']:.3f} |",
        f"| p50 latency (s) | {summary['p50_latency_s']:.2f} |",
        f"| p95 latency (s) | {summary['p95_latency_s']:.2f} |",
        f"| input tokens | {summary['total_input_tokens']} |",
        f"| output tokens | {summary['total_output_tokens']} |",
        f"| total cost (USD) | {summary['total_cost_usd']:.4f} |",
        f"| $/query | {summary['cost_per_query_usd']:.4f} |",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())