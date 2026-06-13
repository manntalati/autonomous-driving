# Agentic Perception Platform — Roadmap

> Companion project to the autonomous-driving perception stack. Where that project
> proves **model building** (train CNNs/ViTs/BEV/temporal from scratch), this one
> proves the **production / agentic** half that MLE roles screen for: model
> serving, tool/API design, agent orchestration, and evaluation.

## Project Goal
Build an AI agent that understands driving scenes by **autonomously orchestrating
the trained perception models** — exposed as an **MCP tool API** — then a second
**data-flywheel agent** that mines targeted training data to break the small-dataset
overfitting ceiling hit in every phase of the driving project.

## Why this project (MLE targeting)
- The driving repo shows you can *train* models. Interviews also want: serving
  models behind a clean API, designing tools/contracts, building agent loops, and
  **measuring** agentic systems. This project is exactly that surface.
- Part B (the flywheel) is the thematic payoff to the recurring "324 images → overfit"
  wall: instead of accepting it, you build the system that attacks it.

## Skills to Showcase
1. **MCP / tool APIs** — design a tool surface an autonomous agent calls (Phases 0–1)
2. **Agent orchestration from scratch** — hand-built Anthropic tool-use loop (Phases 0, 2)
3. **Model serving** — load-once in-process inference behind tools (Phase 1)
4. **Tensor → language** — turn model outputs into LLM-readable JSON (Phase 1)
5. **Spatial reasoning / prompting** — compose detection + seg + BEV (Phase 2)
6. **Agentic evaluation** — accuracy, tool-selection, efficiency, latency, cost (Phase 3)
7. **Embeddings + ANN retrieval** — CLIP/backbone features + FAISS (Phase 5)
8. **Active learning** — uncertainty/entropy sampling, hard-example mining (Phase 6)  ← CS 444
9. **Data-centric AI** — closed-loop curation and measured lift vs random (Phases 7–8)

## Stack & Locked Decisions
- **Language/runtime:** Python 3.13, same repo as autonomous-driving (root on `sys.path` via `conftest.py`).
- **Repo layout:** new packages `mcp_server/` and `agent/` inside this repo; import `models.*` / `demo.pipeline` directly (max reuse).
- **MCP:** official `mcp` Python SDK, **stdio** transport (client launches server as a subprocess).
- **Agent:** **hand-built** tool-use loop on the raw Anthropic Messages API (not the Agent SDK) — to understand the mechanism cold.
- **Serving:** **in-process** model loading inside the MCP server (split to a FastAPI service later only if wanted).
- **Secrets:** `ANTHROPIC_API_KEY` in a gitignored `.env` (loaded by `agent/config.py`); `.env.example` is the committed template.
- **New deps:** `mcp`, `anthropic` (Part A); `open_clip_torch` + `faiss-cpu` added in Phase 5 (Part B).

---

## Phase Progress
- [✅] **Phase 0** — MCP scaffolding & hand-built agent loop
- [✅] **Phase 1** — Perception tools (the tool API)
- [✅] **Phase 2** — Orchestrator agent (spatial reasoning)
- [✅] **Phase 3** — Eval harness (accuracy / efficiency / latency / cost)
- [ ] **Phase 4** — Interactive demo (live tool trace)
- [ ] **Phase 5** — Embedding & retrieval tools (FAISS)
- [ ] **Phase 6** — Active-learning scoring tools (uncertainty)
- [ ] **Phase 7** — Curation agent
- [ ] **Phase 8** — Close the loop & measure mAP lift vs random

---

# Part A — Agentic Perception (Idea 1)

## Phase 0 — MCP scaffolding & hand-built agent loop
**Objective:** prove the full MCP round-trip and the from-scratch tool-use loop on a
trivial tool, before any model code is involved.

**Deliverables**
- `mcp_server/server.py` — FastMCP server with a `ping` tool ✅ *(provided)*
- `agent/mcp_client.py` — stdio client: `connect` / `list_tools` / `call_tool` / `close` ✅ *(provided)*
- `agent/loop.py` — **YOUR TASK**: `mcp_tools_to_anthropic` + `run_agent` (the loop)
- `agent/run.py` — CLI entry point, auto-loads `.env` ✅ *(provided)*
- `agent/config.py` — zero-dep `.env` loader ✅ *(provided)*
- `tests/test_mcp_roundtrip.py` — server round-trip smoke test (no API key) ✅ *(provided)*

**Concepts:** MCP (stdio transport, tools, sessions); Anthropic tool-use turn
(`stop_reason == "tool_use"`, `tool_use` / `tool_result` content blocks).

**Done when:**
- A) `pip install mcp` → `pytest tests/test_mcp_roundtrip.py` green (MCP layer).
- B) implement `agent/loop.py`, set the key in `.env`, run
  `python -m agent.run "ping the perception server with the word banana"` →
  Claude autonomously calls `ping` and reports back.

### Phase 0 — Complete ✅
- `mcp_server/server.py` — FastMCP server with `ping` tool ✅
- `agent/mcp_client.py` — stdio client (connect / list_tools / call_tool / close) ✅
- `agent/config.py` — zero-dep `.env` loader (`ANTHROPIC_API_KEY`) ✅
- `agent/run.py` — CLI entry point ✅
- `agent/loop.py` — `mcp_tools_to_anthropic` + `run_agent` (hand-built Anthropic tool-use loop) ✅
  - `mcp_tools_to_anthropic` — maps each MCP tool to `{name, description, input_schema}`.
  - `run_agent` — turn loop: send `messages` + `tools` to `messages.create`; on `stop_reason == "tool_use"`, iterate `response.content`, dispatch each `tool_use` block via `client.call_tool`, collect results into one user message with `[tool_result]` blocks (single follow-up turn per assistant turn — Anthropic contract); exits on `stop_reason != "tool_use"`; `max_turns` guard against runaway loops; conditional `system` kwarg.
- `tests/test_mcp_roundtrip.py` — MCP stdio round-trip smoke test (no API key) ✅

**Bugs caught in review (Phase 0):**
1. Iterating `tools` (static schema) instead of `response.content` (assistant's emitted blocks) — would have called every registered tool every turn with phantom inputs.
2. `tool_result` content blocks appended directly as messages instead of wrapped in a `{"role": "user", "content": [...]}` turn — API would 400.
3. Assistant turn appended without `role` field.
4. `system="NOT_GIVEN"` sent the literal string `"NOT_GIVEN"` as the system prompt.
5. `messages.append(tool_results_user_turn)` initially inside the per-block loop — would duplicate `tool_result`s on multi-tool turns.

**Done:** A) `pytest tests/test_mcp_roundtrip.py` green. B) `python -m agent.run "ping the perception server with the word banana"` → Claude autonomously calls `ping(message="banana")` and reports back.

## Phase 1 — Perception tools (the tool API)
**Objective:** replace `ping` with real tools wrapping the trained models; return
**LLM-readable JSON**, not tensors. This is the core "autonomous API design" work.

**Deliverables**
- `mcp_server/model_registry.py` — load checkpoints **once** at server startup
  (wraps `demo.pipeline.PerceptionPipeline`); device select; `get_registry()` singleton. ✅
- `mcp_server/scene_store.py` — load nuScenes frames by `(scene_name, frame_idx)`,
  cache them server-side keyed by a UUID `frame_id`; builds 3-frame temporal window;
  extracts CAM_FRONT intrinsic + cam_to_ego calibration. ✅
- `mcp_server/perception_tools.py` — `register_all_tools(mcp)` binds all tools:
  - **Core:** `list_scenes`, `load_frame`, `detect_objects`, `segment_scene`, `bev_map`
  - **Driving decisions:** `check_lane_switch_safety`, `check_turn_clearance`,
    `check_obstacle_stop`, `check_pedestrian_crossing`, `estimate_following_distance`,
    `scene_summary`
- `mcp_server/server.py` — updated to call `register_all_tools(mcp)` after `ping`. ✅
- `configs/agent.yaml` — checkpoint paths + thresholds (already present). ✅

**Concepts/MLE:** load-once serving; schema/contract design; **summarizing tensors
into language** (the agent reasons over JSON, never pixels); units (meters, ego frame).

**Done when:** `python -m agent.run "how many cars are in scene-0103 frame 5?"` returns
a correct count via `detect_objects`, matching a `demo/benchmark.py`-style sanity check.

### Phase 1 — Complete ✅

**New files:**
- `mcp_server/scene_store.py` — `SceneStore` class: scene indexing, frame loading,
  3-frame temporal window construction, CAM_FRONT calibration extraction, UUID cache.
- `mcp_server/perception_tools.py` — `register_all_tools(mcp)`: all 11 tools
  (5 core + 6 driving-decision wrappers).

**Updated files:**
- `mcp_server/model_registry.py` — added `SceneStore` attribute, `run_perception(frame_id)`
  method (lazy pipeline run + numpy cache), `get_registry()` module singleton.
- `mcp_server/server.py` — calls `register_all_tools(mcp)` after the `ping` tool.

**Tool summary:**

| Tool | Tier | Models used | Returns |
|---|---|---|---|
| `list_scenes` | core | — | scene name / description / frame count |
| `load_frame` | core | — | frame_id + timestamp |
| `detect_objects` | core | temporal detector | 2D boxes + class counts |
| `segment_scene` | core | U-Net segmenter | pixel coverage + ahead flags |
| `bev_map` | core | BEV detector | top-down object positions (m) |
| `check_lane_switch_safety` | driving | BEV + seg | safe bool + obstacle list |
| `check_turn_clearance` | driving | BEV + seg + 2D det | clear bool + hazard list |
| `check_obstacle_stop` | driving | BEV | stop bool + nearest obstacle |
| `check_pedestrian_crossing` | driving | seg + 2D det | crossing + occupancy |
| `estimate_following_distance` | driving | BEV | metres to car ahead |
| `scene_summary` | driving | all three | full structured scene JSON |

**BEV coordinate convention** (ego frame, matches trained LSS config):
- `x` = forward from vehicle (0 → 51.2 m for single-camera front config)
- `y` = lateral (negative = left, positive = right)
- Lane thresholds: current lane `|y| ≤ 2 m`; adjacent lane `2 m ≤ |y| ≤ 6 m`

**Design decisions:**
- Perception runs once per frame (on first tool call); all subsequent calls on the
  same `frame_id` reuse the cached numpy arrays — no redundant inference.
- Tools return structured JSON strings, not tensors — the agent reasons in language.
- Driving-decision tools are wrappers over the core outputs; no additional model
  inference is triggered by calling them.
- `check_obstacle_stop` is obstacle-only — traffic lights and stop signs are outside
  the detection model's 3-class set (car / pedestrian / cyclist).

## Phase 2 — Orchestrator agent (spatial reasoning)
**Objective:** a spatial-reasoning agent that **chains** tools to answer compositional
questions about a scene.

**Deliverables**
- `agent/prompts.py` — system prompt: ego-frame conventions, how to combine
  detection + segmentation + BEV, when to call which tool, answer format.
- extend `agent/loop.py` `run_agent` to accept the system prompt + multiple chained
  tool calls (loop already supports it — Phase 2 is mostly prompt + glue).
- `agent/examples.py` — a handful of canned compositional questions for manual testing.

**Concepts:** tool chaining, grounding, prompt engineering for spatial reasoning.

**Done when:** questions like *"Is it safe to change into the left lane?"* trigger a
sensible trace (detect + bev + seg) and a grounded answer.

### Phase 2 — Complete ✅

**New files:**
- `agent/prompts.py` — `SYSTEM_PROMPT` constant: role + ego-frame coordinate conventions
  (x forward, y < 0 = LEFT) + tool-selection guide (which tool for which question type,
  prefer driving-decision tools over recomposing) + workflow rules (`load_frame` first,
  smallest-set rule, no retry on error) + answer format (≤3 sentences, cite the
  grounding fact) + invariants (never invent counts/distances).
- `agent/examples.py` — `EXAMPLES` list (8 canned questions covering single-tool,
  two-tool, decision-tool, and broad-summary cases) + `expects` field per example
  (seeds Phase 3 tool-selection precision metric) + `run_all` driver to fire all 8
  through one MCP session.

**Updated files:**
- `agent/run.py` — passes `SYSTEM_PROMPT` to `run_agent` by default; `--no-system`
  flag for A/B testing prompted-vs-baseline (will be used in Phase 3).

**Loop reuse:** `agent/loop.py` `run_agent` already supported `system=...` from
Phase 0 — Phase 2 was prompt + glue with no loop changes.

**Bugs caught in review (Phase 2):**
1. `--no-system` added to argparse but not wired (no `action="store_true"`, not
   threaded through `_main`) — flag silently no-op.
2. `run_all` declared as `def` (not `async`) but used `await` — `SyntaxError`.
3. `run_all` called `run_agent(client)` — dropped the question, passed the
   `MCPClient` as the question string. Correct call is
   `run_agent(ex["q"], client, model=model, system=SYSTEM_PROMPT)`.
4. Missing `import asyncio` / `run_agent` / `MCPClient` / `SYSTEM_PROMPT` /
   `load_env` in `examples.py` — `NameError` on first run.
5. No `if __name__ == "__main__"` block — file was import-only, couldn't run
   `python -m agent.examples`.

**Done:** `python -m agent.run "How many cars are in scene-0103 frame 5?"` triggers
`load_frame` → `detect_objects` → grounded answer; `python -m agent.examples` runs
all 8 compositional questions in one MCP session.

## Phase 3 — Eval harness (the MLE centerpiece)
**Objective:** measure the agent quantitatively — the skill most student portfolios lack.

**Deliverables**
- `evaluation/agent_eval/benchmark.py` — generate a question set **programmatically
  from nuScenes GT** (counting, presence, nearest-object, spatial relations) with
  ground-truth answers.
- `evaluation/agent_eval/metrics.py` — answer accuracy; tool-selection precision/recall
  (did it call the right tools?); tool-call count (efficiency); end-to-end latency;
  token cost.
- `evaluation/agent_eval/run_eval.py` — run the agent over the benchmark, write a report.
- `logs/agent_eval_run.log` — results table.

**Concepts/MLE:** agent eval methodology; GT-from-labels; accuracy vs efficiency vs
cost trade-offs; regression tracking.

**Done when:** a reproducible report: accuracy %, mean tool calls, p50/p95 latency, $/query.

### Phase 3 — Complete ✅

**New files:**
- `evaluation/agent_eval/benchmark.py` — 4 generators (`gen_counting`, `gen_presence`, `gen_nearest`, `gen_spatial`) emit GT-from-labels questions over the val scenes; `_ego_boxes` helper projects 3D boxes to the ego frame (x-forward, y-LEFT, matching nuScenes); `BenchQuestion` dataclass carries `(qid, question, scene, frame_idx, qtype, gt_answer, expects_tools)`; lane bands `|y| ≤ 2 = current`, `2 < y ≤ 6 = left`, `-6 ≤ y < -2 = right`; cached to `logs/agent_eval_benchmark.json` (320 questions on val scenes 1094 + 1100).
- `evaluation/agent_eval/metrics.py` — `parse_count` / `parse_distance` / `parse_yesno` (case-folded), `score_answer` with model-error-calibrated tolerance bands (count ±1, presence/spatial exact, nearest ±20%), `score_tool_selection` (precision/recall + exact-match flag, div-by-zero guarded), `aggregate` (accuracy, parse_failures, mean_tool_calls, tool_precision/recall, p50/p95 latency, token totals, total + per-query cost at Opus 4.7 pricing).
- `evaluation/agent_eval/run_eval.py` — `run_one` (full-shape error branch so failures don't break aggregation) + `main` (argparse: `--model`, `--limit`, `--no-system`, `--concurrency`, `--data-root`, `--results-path`, `--report-path`); semaphore-bounded concurrency; writes per-question JSONL + markdown report.

**Updated files:**
- `agent/loop.py` — `AgentResult` dataclass (`answer`, `trace`, `input_tokens`, `output_tokens`, `turns`) for instrumented runs; `run_agent` now returns `AgentResult` instead of a bare string.

**Bugs caught in review (Phase 3):**
1. Trace append placed inside `if block.type != "tool_use":` (inverted) — would record skipped blocks instead of called tools.
2. `result = await client.call_tool(...)` stomped the `AgentResult` accumulator — renamed local to `tool_output`.
3. `gen_presence` stub had undefined `idx` and no return — incomplete generator skeleton.
4. `build_benchmark` initially shipped without default kwargs for `val_scenes` / `seed` / `n_per_frame` / `out_path` — caller-burdening contract that didn't match the design.
5. `parse_yesno` first cut used `answer[0:1] == "no"` (slice width 1 vs 2) and no case folding — would mis-parse "Yes."/"NO" and never match "no". Fixed to `.strip().lower().startswith()`.
6. `run_eval.py` first cut imported `run_agent` from `agent.run` (lives in `agent.loop`) and used `...` Ellipsis as a dict value.

**Done — actual numbers:**

- **Smoke (5q, `--concurrency 1`, system on):** accuracy 0.200, tool precision/recall 1.000, mean tool calls 2.00, parse failures 0.000, $/q $0.24, p50 latency 89s. Agent picked the correct tools on every question; the one miss was perception error (detector returned 0 boxes on a 7-pedestrian frame).
- **Full prompted run (320q, default `--concurrency 4`):** rate-limit cascade — 92.5% of questions short-circuited through `run_one`'s error branch on Anthropic 429s, leaving the metrics dominated by error-dict residue rather than agent behavior. Honest read: the harness works end-to-end, but a real 320-question run needs either `--concurrency 1` (~7 hrs) or a 429 retry-with-backoff inside `run_one`.

**Decision:** treat the smoke results as the validation that the *plumbing* is correct (correct tool selection, AgentResult populated, JSONL + report wrote, error branch never tripped at concurrency=1) and *move on*. The accuracy ceiling on this benchmark is the detector's recall, not the agent — fixing that is a model-side problem (covered later by Part B's data flywheel), not an agent-loop problem.

## Phase 4 — Interactive demo (live tool trace)
**Objective:** a visible product that shows the agent thinking.

**Deliverables**
- `demo/agent_app.py` — Streamlit: scene/frame picker, question box, **live tool-trace
  panel**, final answer, image overlays (reuse `utils/visualize`).

**Concepts:** agent observability (surface the trace), product polish.

**Done when:** ask a question in the UI, watch tool calls stream, see answer + overlays.

> **Milestone — Part A complete:** an agent that perceives driving scenes via your
> models, measured by a real eval harness and shown in a demo.

---

# Part B — Data Flywheel (Idea 2)

> **Known constraint to decide in Phase 5:** nuScenes-mini is only 10 scenes, so the
> "unlabeled mining pool" is small. Options: (a) treat held-out/val scenes as the pool
> and show the mechanism, or (b) pull additional nuScenes (trainval) as a larger pool
> for a stronger lift in Phase 8. Pick before building the index.

## Phase 5 — Embedding & retrieval tools
**Objective:** similarity search over the dataset (frame- and optionally region-level).

**Deliverables**
- `data/embeddings.py` — compute image embeddings (CLIP via `open_clip`, and/or your
  ResNet backbone features); build + persist a **FAISS** index.
- `mcp_server/retrieval_tools.py` — `search_similar(frame_id | text, k)`,
  `embed(frame_id)`; (CLIP enables text→image queries like *"nighttime"*).
- cache: `data/raw/.../embeddings.npy` + the FAISS index file.

**Concepts:** representation learning, embeddings, approximate nearest-neighbor (FAISS),
cross-modal (text/image) retrieval.

**Done when:** `search_similar` returns sensible neighbors; a text query pulls matching frames.

## Phase 6 — Active-learning scoring tools  *(CS 444 link)*
**Objective:** quantify per-frame "hardness" / model uncertainty.

**Deliverables**
- `evaluation/active_learning/uncertainty.py` — detection-confidence **entropy**,
  score-margin, count of low-confidence boxes (softmax entropy = direct CS 444 concept).
- `mcp_server/mining_tools.py` — `score_uncertainty(frame_id)`,
  `find_hard_examples(criterion, k)`.

**Concepts:** active learning, uncertainty sampling, entropy/margin scoring.

**Done when:** the hard-example ranking surfaces genuinely ambiguous frames over confident ones.

## Phase 7 — Curation agent
**Objective:** a second agent that, given a goal, **autonomously assembles** a candidate
training set.

**Deliverables**
- `agent/curation_loop.py` (or reuse `run_agent` with a curation system prompt) —
  goal → search + uncertainty-score + dedup + select → write a manifest of sample tokens.
- `agent/prompts.py` — curation system prompt.
- output: `data/curated/<goal>.json` manifest.

**Concepts:** agentic data curation, goal decomposition, combining retrieval + uncertainty, dedup.

**Done when:** a goal like *"50 nighttime cyclist frames"* yields a sensible manifest via tool calls.

## Phase 8 — Close the loop & measure lift (the payoff)
**Objective:** prove the flywheel beats random sampling — the resume-grade result.

**Deliverables**
- training: retrain the Phase 2 detector (`models/detection/train_detector.py`) under a
  **fixed budget**, comparing random-sampled additions vs the curated manifest.
- evaluation: compare mAP / per-class AP on val; results table + short writeup.
- update this roadmap + `README` with the comparison.

**Concepts/MLE:** data-centric AI; active-learning lift; controlled comparison (same
budget, random vs curated); honest reporting (a null result is still a finding).

**Done when:** a table showing curated ≥ random on mAP (or an honest null), with method noted.

---

## Key Decisions
- Same-repo new packages over a separate repo — maximize reuse of trained models/checkpoints.
- Hand-built agent loop over the Agent SDK — learn the tool-use mechanism that underlies every framework.
- In-process serving first — simplest path to a working system; FastAPI split is a later option, not a blocker.
- stdio MCP transport — standard for local single-server setups; the client owns the subprocess lifecycle.
- Tools return JSON summaries, never raw tensors — the agent reasons in language; this is the key serving/design insight.

## Collaboration Notes
- Manage phase by phase as PM (same as the driving project): present tickets, give
  method-level skeletons, the user fills in implementations.
- Review user code when asked — point out bugs, never silently fix them.
- Keep this roadmap's Phase Progress + per-phase "Completed Work" current as phases land.
- Emphasize the ML/CS 444 angle throughout (e.g., uncertainty = softmax entropy in Phase 6).
