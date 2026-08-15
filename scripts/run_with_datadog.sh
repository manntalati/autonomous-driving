#!/usr/bin/env bash
# Run any entry point with Datadog LLM Observability enabled.
#
#   bash scripts/run_with_datadog.sh python -m agent.run "how many cars in scene-0103 frame 5?"
#   bash scripts/run_with_datadog.sh streamlit run demo/showcase.py
#
# Config comes from .env (gitignored). This script only checks that DD_API_KEY is
# set before spending time on a run that would silently not be traced.
set -u
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a

if [ -z "${DD_API_KEY:-}" ]; then
  echo "DD_API_KEY is not set. Add it to .env (which is gitignored):"
  echo "    DD_API_KEY=<key from Datadog -> Organization Settings -> API Keys>"
  exit 1
fi

export DD_LLMOBS_ENABLED="${DD_LLMOBS_ENABLED:-1}"
export DD_LLMOBS_ML_APP="${DD_LLMOBS_ML_APP:-autonomous-driving}"
export DD_LLMOBS_AGENTLESS_ENABLED="${DD_LLMOBS_AGENTLESS_ENABLED:-1}"
export DD_SITE="${DD_SITE:-datadoghq.com}"

echo "Datadog LLM Observability: ml_app=$DD_LLMOBS_ML_APP site=$DD_SITE"
echo "Traces: https://app.datadoghq.com/llm/traces"
echo ""

# ddtrace-run wraps the process for CLI entry points. The Streamlit apps also
# call enable_llm_observability() in-process (agent/loop.py), so they are traced
# either way — running them through here just makes the intent explicit.
exec .venv/bin/ddtrace-run "$@"
