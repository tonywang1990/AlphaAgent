# AlphaAgent

A general-purpose agent for optimization problems of one shape: data split into train / val / test, a
clearly defined goal and eval function, and code (a heuristic, a model, or a combination of components)
that must score well on the sealed held-out test.

## Contents

| Path | What it is |
|---|---|
| `docs/case_studies/` | Two worked examples of the loop: DOTA (trading-signal ensemble search) and adsbench (pairwise ad-reach prediction) |
| `reports/Research agents for optimization.md` | Survey of existing research agents and harnesses, ten design axes, five candidate frameworks, a task contract, harness invariants, a first build and experiments |
| `research_notes/Research agents for optimization/` | The cited research notes behind the report, one file per research family |
