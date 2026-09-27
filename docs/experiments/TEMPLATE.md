# R?: <experiment title>

Copy this file to `docs/experiments/R<n>_<short_name>.md` and fill in everything
above **Result** before the first training run. Once the status is `registered`,
the sections above Result are frozen. Any later change is added under
**Amendments** with a date and a reason, and is never edited in place.

| | |
|---|---|
| **Status** | draft / registered / running / done / abandoned |
| **Question** | Q1 (can it tell when it is wrong) / Q2 (data, representation or geometry) / Q3 (far-range depth) / Q4 (acting on uncertain perception) |
| **Registered** | YYYY-MM-DD, git `<sha>` |
| **Time-box ends** | registered + 2 weeks (rule 5) |
| **Owner** | |

## Baseline

The fixed reference this experiment is compared against (rule 7).

| | |
|---|---|
| Scorecard row | `results/scorecard/<baseline>.json` |
| Config | `configs/<...>.yaml`, hash `<config_hash>` |
| Checkpoints | `checkpoints/<...>_s{seed}.pt`, seeds `0,1,2` |
| Git SHA | `<sha>` |

## The one change

The single variable this experiment changes. Everything else stays identical to
the baseline config. List every config key that differs; there should be one, or
one coherent group, such as a backbone and the adapter it needs.

## Hypothesis

One falsifiable sentence, with a direction and a size.

## Metric and threshold

Every metric is a scorecard cell (`evaluation/scorecard.py`, `CELL_ORDER`). No
one-off eval scripts (rule 1).

| Role | Cell | Baseline | Threshold |
|---|---|---|---|
| Primary | `det....` | | |
| Guard | `det....` | | must not ... |
| Reported, not decided on | `...` | | |

**Spread (rule 2).** A result counts only if it clears the baseline's 95%
interval: for an improvement, the experiment's mean lies beyond the baseline's
upper bound (lower bound for a cell where smaller is better). A mean inside the
baseline's interval is no effect, whatever its sign.

## Failure condition

What result refutes the hypothesis, stated so that it can be checked mechanically
against the scorecard row. Also state what counts as **inconclusive**, for
example real but below threshold, or interval too wide to decide.

## Protocol

- **Seeds:** at least 3 (rule 2). List them.
- **Reproduce first (rule 4):** for a method from a paper, the number you will
  reproduce in the paper's own setting before porting it, and how close counts as
  reproduced.
- **Training:** config, epoch budget, early stopping or fixed epochs, and which
  checkpoint is scored (best or last).
- **Scoring:** the exact `python -m evaluation.scorecard ...` command.
- **Compute budget:** the estimated time per seed, and the device.

## Known confounders

What else could produce the same result, and how the design rules it out or at
least names it.

---

## Result

*(filled in after the run; the sections above stay frozen)*

Scorecard rows: `results/scorecard/<name>.json` (git `<sha>`)

| Cell | Baseline mean [95% CI] | This experiment mean [95% CI] | n |
|---|---|---|---|
| | | | |

## Verdict

confirmed / refuted / inconclusive. Give one paragraph, and apply the failure
condition exactly as written above. Negative results get written up in the same
format as wins (rule 6).

README write-up: link to the phase section.

## Amendments

Dated changes to anything above **Result**, made after registration, each with
its reason.
