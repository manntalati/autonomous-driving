"""
Datadog LLM Observability wiring.

WHAT GETS TRACED
----------------
Only the agent's Anthropic calls. Every LLM request in this project funnels
through `agent/loop.py:run_agent`, so that one call site is the entire surface
Datadog sees — detection, segmentation, BEV, radar and the trust layer all run
locally and never leave the machine.

TWO WAYS TO ENABLE IT
---------------------
1. `ddtrace-run <command>` — the documented path, good for CLI entry points:

       ddtrace-run python -m agent.run "how many cars are in scene-0103 frame 5?"

2. `enable_llm_observability()` — call it in-process. This is what the Streamlit
   apps use, because `ddtrace-run` wraps the *process*, and Streamlit re-executes
   the script body on every widget interaction inside one long-lived process. The
   wrapper still works there, but in-process init is explicit about when tracing
   starts and is idempotent across reruns.

Both read the same environment variables, so there is one place to configure:

    DD_LLMOBS_ENABLED=1              turn it on (0 or unset disables cleanly)
    DD_LLMOBS_ML_APP=autonomous-driving
    DD_API_KEY=...                   from Datadog → Organization Settings → API Keys
    DD_SITE=datadoghq.com            or datadoghq.eu, us3.datadoghq.com, ...
    DD_LLMOBS_AGENTLESS_ENABLED=1    send directly, no local Datadog agent needed

NEVER hardcode DD_API_KEY. It lives in `.env` (gitignored) and is read from the
environment — a key committed to a repo is a key that has to be rotated.
"""
from __future__ import annotations

import os
from typing import Optional

_ENABLED: Optional[bool] = None      # cached so reruns do not re-initialise


def llmobs_enabled() -> bool:
    """True when tracing is configured and switched on."""
    return os.environ.get("DD_LLMOBS_ENABLED", "0") not in ("0", "", "false", "False")


def enable_llm_observability(verbose: bool = False) -> bool:
    """
    Start LLM Observability if it is configured. Returns True when tracing is on.

    Safe to call repeatedly — the result is cached, so Streamlit reruns do not
    stack up initialisations.

    Deliberately non-fatal: observability must never take down the thing it is
    observing. A missing key, an unreachable Datadog, or an incompatible ddtrace
    version produces a warning and a False, not an exception in the middle of a
    user's drive.
    """
    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED

    from agent.config import load_env
    load_env()                                    # pick up .env before reading vars

    if not llmobs_enabled():
        _ENABLED = False
        return False

    if not os.environ.get("DD_API_KEY"):
        if verbose:
            print("[llmobs] DD_LLMOBS_ENABLED=1 but DD_API_KEY is empty — tracing off.")
        _ENABLED = False
        return False

    try:
        from ddtrace.llmobs import LLMObs

        LLMObs.enable(
            ml_app=os.environ.get("DD_LLMOBS_ML_APP", "autonomous-driving"),
            api_key=os.environ["DD_API_KEY"],
            site=os.environ.get("DD_SITE", "datadoghq.com"),
            agentless_enabled=os.environ.get("DD_LLMOBS_AGENTLESS_ENABLED", "1") not in ("0", ""),
            integrations_enabled=True,            # auto-instruments the anthropic SDK
        )
        _ENABLED = True
        if verbose:
            print(f"[llmobs] tracing to ml_app="
                  f"{os.environ.get('DD_LLMOBS_ML_APP', 'autonomous-driving')}")
    except Exception as e:                        # noqa: BLE001 — see docstring
        print(f"[llmobs] disabled — {type(e).__name__}: {e}")
        _ENABLED = False
    return _ENABLED


def annotate_agent_run(question: str, result, scene: Optional[str] = None) -> None:
    """
    Attach project-level context to the current LLM span.

    The SDK integration already records the model, tokens, latency and the raw
    messages. What it cannot know is what the call was *for*, so this adds the
    things worth filtering and grouping on in the Datadog UI:

      - which perception tools the agent chose (the interesting behaviour — the
        agent picks its own tools, so the trace is a record of its reasoning path)
      - whether it exhausted its turn budget, i.e. returned a degraded answer
      - which scene it was asked about

    A no-op when tracing is off, so callers need no conditional.
    """
    if not _ENABLED:
        return
    try:
        from ddtrace.llmobs import LLMObs

        LLMObs.annotate(
            input_data=question,
            output_data=getattr(result, "answer", ""),
            tags={
                "scene": scene or "unknown",
                "exhausted": str(getattr(result, "exhausted", False)),
                "n_tool_calls": str(len(getattr(result, "trace", []))),
                "tools": ",".join(getattr(result, "trace", [])) or "none",
            },
            metrics={
                "turns": getattr(result, "turns", 0),
                "input_tokens": getattr(result, "input_tokens", 0),
                "output_tokens": getattr(result, "output_tokens", 0),
            },
        )
    except Exception as e:                        # noqa: BLE001
        print(f"[llmobs] annotate failed — {type(e).__name__}: {e}")
