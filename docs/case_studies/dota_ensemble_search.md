# Case study: DOTA, an ensemble search for a trading signal

*A learning and optimization problem for a self-improving agent. Written 2026-10-23 from the
LU campaign (ADR 0452, PR #452). Each number here is a measured result from that campaign.*

---

## 1. The problem in one paragraph

We trade one futures contract (LU, low-sulphur fuel oil on the INE exchange). A *signal* is a
small model that predicts the next few seconds of price move from the order book. Production
runs an *ensemble*: a weighted sum of up to 12 signals. The agent's job is to **choose which
signals to train, and which of them to combine**, so that the ensemble earns more on days
nobody has shown it. Training a signal costs money and time, so the agent has a fixed
**budget of models it may train**. It must learn from the val days and be judged on held-out
test days, and it must do better than two simple baselines (random picks, and a
Bayesian-optimization "surrogate") at every budget. The setting is *online*: the agent runs
campaign after campaign on the same symbol, and should get better at the task each time.

In the generic template:

| Template slot | DOTA |
|---|---|
| Data | per-tick order-book features and forward returns, ~1 year, in 3 market periods |
| Train set | the *fit* days (fit the ensemble weights) + the *val* days (choose models and sets) |
| Test set | the *test* days (alternate eval days, sealed), plus a final *oos* block never touched |
| Decision | which recipes to train (≤ budget), which ≤12 trained models form the ensemble |
| Eval function | EV of the ensemble's top-1% predictions, averaged over 10s/30s/60s and 3 periods, as a ratio to prod |
| Baselines | random recipes; surrogate (random forest + Thompson sampling) |
| Success | beat both baselines on test at every budget, with t ≥ 2; score should rise with budget |

---

## 2. Domain background (the minimum you need)

- **Tick data.** The exchange sends a book snapshot every 500 ms. Every snapshot gives
  one row: hundreds of *features* (order imbalance, trade flow, spreads, prices of
  related contracts) and *labels* (the forward return over 1s, 5s, 10s, 30s, 60s, 180s).
- **A signal (member, model)** is fitted on past days and outputs a prediction per row. It is
  fully named by its *recipe*, a dotted string of choices, one per axis:

  ```
  arch . horizon . target . feature_set . selection . lookback . leg . session . seeds . sN
  tree . 10s     . y      . lu2f0_tree10 . all      . 40d      . hot . split   . x1    . s1
  ```

  | Axis | Values (LU) | Meaning |
  |---|---|---|
  | arch | tree, linear, linear_b, ridge, gru, lstm, … (8 declared, 3 used) | model family; `linear_b` = bounded linear |
  | horizon | 1s, 5s, 10s, 30s, 60s, 180s | which forward return it is trained on |
  | target | y, yskip, r2f0 | plain return; return skipping the first tick; residual of prod |
  | feature_set | 22 sets (agg_all, agg_xs, frozen194, lu2f0_*, …) | which columns it sees |
  | feature_selection | all, s1, s10, s30, s60 | screening of the set |
  | lookback | 40d, 80d, 120d | training window |
  | leg / session | hot/slow, pooled/split | serving latency class; day+night pooled or not |

  The grammar has laws, for example "no tree on an unscreened aggregated pool". In practice
  the legal and buildable LU recipes number in the hundreds. The pre-fitted benchmark universe
  holds **643** of them (276 trees, 197 linear_b, 170 linear, across 4 horizons).
- **The ensemble (F)** is at most `k_max = 12` members. Its weights are not free parameters
  the agent sets. They are a fixed function of the member set: a least-squares fit on the
  *fit* days. So choosing F means choosing a **set**.
- **Prod** is the incumbent LU ensemble: 2 members (a 1s linear at weight 1.0 and a 10s tree at
  weight 0.5). Every score is reported as a ratio to prod's score, so 1.0 = prod.
- **Fill-aware EV** (expected value) is the money metric. It is the average forward return,
  in ticks, of the rows where the prediction is in its top 1% by size, signed by the
  prediction. Those are the moments the strategy would actually trade.

---

## 3. Data and splits

### 3.1 Time layout

Market behaviour changes by regime, so the calendar is cut into **three periods**, each
with three consecutive blocks:

| Period | aggregator (fit) | eval (val + test) | oos (sealed) | Daily vol (bp) agg / eval / oos |
|---|---|---|---|---|
| calm | 2025-07-01 → 07-28 | 2025-07-29 → 08-25 | 2025-08-26 → 09-22 | 1.52 / 1.62 / 1.51 |
| storm | 2026-01-29 → 03-05 | 2026-03-06 → 04-02 | 2026-04-03 → 05-06 | 2.30 / **6.56** / 3.92 |
| recent | 2026-06-24 → 07-21 | 2026-07-22 → 08-18 | 2026-08-19 → 09-15 | 3.05 / 2.67 / 3.54 |

- **Signals** (the members) are trained on days *before* each period's aggregator block (a
  40-120 day lookback), so their predictions on these blocks are out of sample for them.
- **fit rows** = the aggregator blocks. Only the ensemble weights are fitted here.
- **eval block** is split by **alternating days within each period**: even days → **val**,
  odd days → **test** (the code calls it `check`). That gives about 10 val and 10 test
  days per period, about 30 of each in all.
- **oos** is outside the data range the loaders will even read. It is the final gate frame
  for a real launch and is never used in this study.

### 3.2 What the agent may see

| Data | Agent | Selection code | Scorer |
|---|---|---|---|
| fit rows (weights) | indirectly (weights are given) | yes | yes |
| val rows (features, labels, F's predictions) | **yes**, in a sandbox, as files | yes | yes |
| test rows | **never**, not even a test score | no | yes |
| oos | never | never | never |

The agent never sees a test number, including in its own ledger. An earlier run (S4a) that
saw "this gain held on test" steered on it. Its result was invalid, because the arms are compared on those very
days.

### 3.3 Two modes

- **Real mode.** "Train" really fits the model, which takes minutes to hours per recipe. A
  bank-mode world of ~800 models took ~43 GB of RAM.
- **Benchmark mode (`--universe`).** All 643 recipes are pre-fitted. "Training" reveals one
  (instant), and the budget is counted the same way. This turns a race that took a day into
  one that takes 20-60 minutes. It is what makes many seeds and many agent episodes possible.
  Feature-set creation is off in this mode, because only pre-fitted models exist.

---

## 4. The goal and the eval function

### 4.1 The objective (as of 2026-10-23): `ev_drift`

```
for each period p in {calm, storm, recent}:
    for each horizon h in {10s, 30s, 60s}:
        EV_p,h = mean over the rows in that period whose |prediction| is in the top 1%
                 of: sign(prediction) * forward_return_h      (gross, in ticks)
    score_p = mean_h EV_p,h  /  (the same quantity for prod in period p)
score = mean_p score_p                        (1.0 = prod; worst_weight = 0)
```

Why the average over 10s/30s/60s: the earlier objective, EV or IC at 10s only, rewarded
**front-loaded** alphas, whose edge is gone by 30s. LU earns as a passive maker that holds
inventory past the fit label, so an edge that decays in seconds does not pay.

### 4.2 Comparing two ensembles: the paired gain

Single scores are noisy, because a day-to-day swing dominates. Every comparison of a new set
against an old one is **paired by day**. Compute both on each day, take the difference,
then report the mean difference and its standard error. A *swap* (replacing a member) or an
agent-proposed ensemble is accepted only when this paired **t ≥ 2 on val**.

### 4.3 The arms' score

For a campaign, record F's test score after each round. Round k is budget 16·k models. Compare the arms at each
budget by the mean test score over seeds or chains, using Welch's t against each baseline.
"Full" budget = 96 models (6 rounds × 16). For headroom, an ensemble picked from all 643
universe models starting from prod reaches **1.128** on test.

### 4.4 The selection objective is not the eval objective

This was the biggest single finding. If F is chosen on the `ev_drift` val days, val rose by
+0.05 to +0.17 while test stayed flat. The top-1% EV on ~30 days is too noisy to select on.
Choosing F on **IC** (correlation with the label, on every row) and scoring on `ev_drift` was
**+0.069 better on test on 6 of 6 banks (t 4.6)**. So: *select on a smooth proxy, score on the
noisy target.* SoM already did this (ADR 0008). It is not universal: on BU, selecting on IC
lost EVf at t −2.6.

---

## 5. What the agent controls

### 5.1 The loop (one campaign = 6 rounds)

```
F = prod members
for round in 1..6:
    select    F = greedy fill + gated swaps over the trained bank, on val   (system, fixed)
    brief     workspace files: F, its diagnosis, the bank, the grammar, the ledger of every
              earlier experiment and lesson, the budget, memory.md, experience.md
    plan      the agent (an LLM session with tools) writes 1-4 experiments    <-- the agent
    realize   experiments -> recipes; unspent budget -> surrogate draws
    train     (or reveal, in benchmark mode)
    re-select F' over the grown bank; adopt the agent's own ensemble if paired t >= 2 on val
    score     F' on test (sealed: written to the run log, never to the agent)
    ledger    one val-only row per experiment + the agent's lessons
```

### 5.2 Actions (the agent's reply)

| Action | What it does |
|---|---|
| `train` | name recipes to train (must be legal and, in benchmark mode, in `available.txt`) |
| `suggest` | spend n models on the surrogate's Thompson draws **inside a region the agent names** (`where: {arch: [linear_b], horizon: [30s]}`) |
| `ensemble` | propose a member set; adopted only at paired t ≥ 2 over the system's pick |
| `feature_set` | compose a new feature set from store columns (screened against random columns first); real mode only |
| `tools` | code the agent wrote (script + test); joins a library for later rounds if its test passes in the sandbox |
| `lessons` | free text, carried to later rounds and (online learning) later campaigns |
| `stop` | end early (ignored in benchmark mode: the budget is fixed) |

### 5.3 Tools (all answer on val days only)

| Tool | Answers |
|---|---|
| `whatif --add/--drop` | F's val score with members changed, and the paired gain + SE |
| `members` | trained models by marginal val gain to F |
| `effects` / `axes` | mean marginal gain per grammar value and value pair, with SE and n |
| `suggest` | the surrogate's top untrained recipes and expected gains |
| `crossval` | honest value of the last batch / whole bank: select on half of val, score on the other half |
| `search` | local search over ensembles from several starts; reports the held-out gain of searching |
| `pcorr` / `screen` | which raw store columns carry signal F lacks |
| `run script.py` | the agent's own Python in a `bwrap` sandbox: val rows only, no network, no other dates |

The agent may write any code it likes, but only in the sandbox. Its file reads go through a
read-only workspace, and every Bash command is audited against an allow-list.

### 5.4 Knowledge the agent carries

- **Offline: cross-symbol memory (`memory.md`).** What the served ensembles on 9 other symbols
  look like, for example "the anchor is a linear on a wide aggregated pool; the second leg is
  a 10s tree; slow linears enter at negative weight". It contains no LU content.
- **Online: the experience store (`experience.md`, `recipes.json`).** One episode per finished
  LU campaign, val-only:
  - the final F;
  - how F's val score grew with the models trained;
  - every recipe's marginal val gain to that final F;
  - which cells paid;
  - what each kind of experiment did;
  - the agent's lessons.

  The next campaign on the same symbol reads it.

---

## 6. The baselines

| Baseline | Algorithm |
|---|---|
| **random** | each round, draw 16 untrained recipes uniformly from what is available |
| **surrogate** | a random forest over the recipe axes (one-hot), fitted on every trained model's marginal val gain to F; pick 16 recipes by Thompson sampling (each pick = the argmax of one random tree) |

Both use the same system selection and the same budget. Neither carries anything between
campaigns, while the agent does (its experience store). A stricter control would be a
surrogate warm-started on the same past episodes. That control has not been run yet.

---

## 7. Results so far (LU, from prod, benchmark mode)

Test score, `ev_drift`, 1.0 = prod (prod's own two members re-weighted score 0.981):

| Models trained | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|
| surrogate (12 seeds) | 0.994 | 1.017 | 1.014 | 1.014 | 1.016 | 1.015 |
| random (12 seeds) | 0.984 | 1.014 | 1.018 | 1.038 | 1.038 | 1.039 |
| agent, episode 1 (memory only; 6 chains) | 1.004 | 1.028 | 1.040 | 1.042 | 1.055 | 1.059 |
| agent, episodes 2-4 (+ own experience; per-chain mean of 6) | **1.054** | **1.054** | **1.060** | **1.063** | **1.063** | **1.062** |
| t vs surrogate / random | 5.6 / 7.8 | 3.5 / 4.1 | 3.5 / 4.4 | 3.5 / 2.2 | 3.2 / 2.2 | 3.3 / 2.0 |
| ceiling: pick from all 643, from prod | | | | | | 1.128 |

Read it this way:

- Experience pays most at small budgets. With 16 models, the experienced agent beats what
  either baseline reaches with 96.
- **The open problem: the curve is flat after 16 models.** The target is a score that keeps
  rising roughly log-linearly with the models trained, toward the 1.128 ceiling.
- Learning is not monotone across episodes. At 96 models, episodes 1, 2, 3, 4 score 1.059,
  1.055, 1.069, 1.062.

Earlier, on the IC objective, a similar picture held. Hand-picking agents without memory
*trailed* the surrogate at every budget. The cross-symbol memory flipped this: the agent
gained 2.3× the surrogate's gain over prod at 24 models (t 4.1).

---

## 8. Challenges (why this is hard)

### 8.1 Noise is larger than most of the effects
- One round's test gain has an SE of ~0.02 on `ev_drift`, about the size of the whole
  difference between arms. Telling arms apart needs 6-12 seeds per arm and paired tests.
- Under that noise, a val gain of +0.001 is usually selection noise. In one measured case a
  swap worth +0.0013 on val scored −0.0016 on test.

### 8.2 Selection overfits val (the winner's curse), at several levels
- **Model level.** With hundreds of trained models, the best on val is partly lucky. Adds
  without a margin raise val and not test.
- **Ensemble level.** A greedy fill plus swaps with no gate lost 0.06 on held-out days. A
  "path switch" (another start's set replacing the incumbent for +0.003 val) lost 0.0185 on test.
  Hence the gates: a swap or a replacement needs paired t ≥ 2.
- **Objective level.** Selecting on the noisy target itself (`ev_drift`) overfit badly. Selecting on IC fixed it (§4.4).
- **Across campaigns (online).** Rebuilding "the earlier final F with the best val score" is itself a max
  over noisy val scores. This is the likely reason episode 4 scored below episode 3.

### 8.3 The search is path dependent and capped
- F is capped at 12 members. Once full, a new model gets in only through a swap that clears
  t ≥ 2. So a bigger bank stops helping, and every arm's curve goes flat at 12 members.
- Where the fill starts matters a lot. From all 643 models: fill from prod → 1.128 test;
  fill from empty → 1.028.
- The gates that protect against noise are the same ones that freeze progress. That trade-off
  sits at the heart of the plateau.

### 8.4 The value of a model depends on the set
- A recipe's worth is its *marginal* gain to the current F, not its standalone score. The same
  model can be top-ranked against one F and useless against another.
- Interactions matter: a value weak on average can be the best cell paired with one other value.
- The surrogate's signal (gain to the current, full F) is weak once F is full. Adding a 13th
  member says little about the gain from a swap.

### 8.5 Budget and cost
- In real mode, a model fit costs minutes to hours and tens of GB of RAM. The agent must
  decide before it sees the result.
- Budget left unspent is lost, so the system fills it with surrogate draws.
- A recipe picked by two experiments was charged twice (a bug, now fixed), which quietly
  handicapped the agent.

### 8.6 Regimes (non-stationarity)
- The three periods differ about 4× in volatility; storm's eval block is 6.6 bp vs 1.6 in calm. A set that wins on
  average can lose a period. Scores are per-period ratios, so one period cannot dominate.
- The train/val/test blocks are adjacent in time, but the regime can still shift between them.

### 8.7 Leakage and honesty constraints
- Labels sit in the same raw files as the features (`forward_return_*`, `yskip_*` columns).
  A tool that builds its column list from the schema must drop them.
- Test must never reach the agent, its ledger, its lessons or its experience store. The earlier
  "honest" `crossval` started from F, which was chosen on all val days, so it was biased. Two
  agents stopped early because of it. It was fixed to start from prod.
- The agent may write code, so the sandbox is the boundary. That covers its file system view
  (val days only), no network, and an audit of every command.

### 8.8 Baseline subtleties that change conclusions
- The surrogate with 2 trained points: all trees predict the same value, so the ties were broken
  by list order and every seed drew the same 16 recipes. Fixed by breaking ties randomly per seed.
- An early random arm had half the budget, and one universe held a recipe family (`r2f0`)
  that could not be built.
- Each of these made one arm look better or worse than it was. Audit the baselines as
  carefully as the agent.

### 8.9 LLM-agent behaviour
- Without priors, an LLM picking by hand chases cells whose val mean is mostly noise, and loses
  to Thompson sampling. The edge came from **knowledge** (other symbols, its own past episodes),
  from **steering** the surrogate (`suggest` with a region), and from **reasoning about the
  ensemble**, not from out-guessing a calibrated sampler recipe by recipe.
- The agent cites its own lessons and evidence well. For example, it declined a searched set
  at t 1.97 whose held-out gain was negative. It also stopped early when its evidence
  said "nothing left", and in benchmark mode stopping is not allowed.
- The agent's brief is part of the system. It is frozen per campaign so arms stay comparable.

### 8.10 Infrastructure
- Memory (43 GB worlds, OOM kills on a shared box), disk (tool states filled 25 GB), process
  concurrency (per-recipe locks; models fitted twice at once collide), and slow EV tools (a
  minutes-long call broke the tool server's socket). All of these needed fixing before the
  science could proceed.

---

## 9. The problem, stated generally

```
Given:   a pool of candidate components C (recipes; each costs c to materialize),
         data split into   FIT (fits the combiner),  VAL (visible, for choices),
                           TEST (sealed, judges),    OOS (never used),
         an incumbent solution S0 (prod),
         an eval function  E(S, days) -> score, noisy, regime-dependent,
         a budget B of components to materialize,
         a combiner: S = argmin least-squares over a chosen subset of <= K components.
Find:    a policy that picks which components to materialize (sequentially, in rounds)
         and which subset to combine, maximizing E(S, TEST) at every budget b <= B,
         using only FIT and VAL.
Online:  the policy faces the same problem repeatedly (campaigns on the same symbol; and
         related problems on other symbols) and may carry knowledge between them; its
         performance should improve with experience and with model count.
```

What makes it a good test problem for a self-improving agent:
1. **A cheap, honest simulator.** Benchmark mode makes one episode take minutes, and test is
   sealed by construction.
2. **Strong, simple baselines** (random; BO-style surrogate) that hand-picking loses to.
3. **The agent may write code** (tools, analyses) under a hard data boundary.
4. **The signal-to-noise is realistic**, so overfitting val is the main failure mode, not
   lack of cleverness.
5. **Transfer is measurable**, from other symbols (offline) and from its own episodes (online).

---

## 10. Open questions (as of 2026-10-23)

1. **Log-linear growth.** Make the score keep rising with the number of models trained,
   instead of plateauing after 16. Suspects: the 12-member cap with the t ≥ 2 swap gate, and
   the rebuild rule. A check is running that re-picks fresh from prod over each run's final bank.
2. **Which rebuild rule** turns past episodes into a starting point without a winner's curse
   (best-val F: 1.069; union of final Fs: 1.072; positive-gain pool: 1.059 on 6 chains).
3. **A warm-started surrogate control,** to show the agent adds more than the data it remembers.
4. **Real mode** (actual fits, feature-set creation) and a second symbol (BU: EVf too noisy to
   separate arms with SE ~0.1).
5. **Selecting on a proxy:** when is IC the right proxy for an EV target, and when not (it
   lost on BU)?

---

## 11. Reference implementation (the best result so far)

The best arm is **agent v9 with cross-symbol memory and its own experience store, selecting on IC
and scoring on `ev_drift`**: §7, episodes 2-4. That is code as of commit `5b9c786d` on branch
`dota-agent-loop`. This section shows how it is built, with the real code (lightly trimmed),
so it can serve as the reference solution for this problem.

### 11.1 How to run it

```bash
cd /qr_scratch/tony/wt/dota-loop && source env/setup.bash
PY=/opt/miniconda3/envs/wuji-dev-env-0.0.1/bin/python
A="--objective ev_drift --select-objective ic --from prod --iters 6 --models 16 \
   --universe /qr_scratch/tony/dota/agent/universe_LU_nor2f0.json \
   --bank /qr_scratch/tony/dota/bank/LU_b1/cache/signal_bank.json"
# one chain = episodes 1..4 in sequence, sharing one experience store
for k in 1 2 3 4; do
  $PY -m dota.agent LU EIc0e$k --policy agent --seed 0 --prompt dota/prompts/agent_v9.md \
      --memory /qr_scratch/tony/dota/agent/memory/cross_symbol_noLU.md \
      --experience /qr_scratch/tony/dota/agent/experience/LU_ei_c0 $A
done
# baselines: the same $A with --policy surrogate|random --seed N (no memory, no experience)
```

Per agent episode: 6 rounds × 1 LLM session (`claude -p`, Opus, effort high, tools
`Read,Grep,Glob,Write,Edit,Bash(./tool:*)`), about 17 minutes wall-clock in benchmark mode.

### 11.2 The architecture

```
                 ┌──────────────── one round ────────────────┐
 bank ──► select F (IC, val) ──► brief workspace ──► LLM agent ──► plan (JSON)
   ▲          │  greedy fill + gated swaps       │ reads files, calls ./tool (val only),
   │          │                                  │ runs its own code in a bwrap sandbox
   │          ▼                                  ▼
   │     surrogate (RF over recipe axes)  ◄── suggest / fill draws
   │                                             │
   └──── train / reveal ◄──── realize experiments (≤ budget) ◄┘
              │
              ▼
     re-select F' (+ agent's ensemble if paired t ≥ 2)  ──► test score (sealed log)
              │
              ▼
     ledger (val only) ──► next round      episode end ──► experience store ──► next campaign
```

Two parts are fixed algorithms that the agent cannot change: how F is chosen from the bank
(§11.3), and the test scorer. Everything the agent decides goes through its plan.

### 11.3 Selection: how F is chosen from a bank (fixed, shared by every arm)

Weights are not chosen. For a member set `jj`, `w = S⁻¹c` is fitted on the FIT rows
(`S` = the members' covariance, `c` = their covariance with the label). The search picks the set.

`dota/env/som/select.py`:

```python
def greedy(ds, fit, cfg, counts, pool, t_min, k_max, start=(), scorer=None, stop_flat=False):
    """adds by the best score while one clears `t_min` (and, `stop_flat`, while it raises it)."""
    cur = list(start)
    while len(cur) < k_max:
        cand = [j for j in pool if j not in cur]
        if not cand:
            break
        s = batch(scorer, [cur + [j] for j in cand])
        best = cand[int(np.nanargmax(s))]
        if stop_flat and cur and not np.nanmax(s) > scorer(cur):
            break                                # the best add lowers val
        cur.append(best)
    return cur


def swap(cur, pool, scorer, rounds, keep=(), gate=None):
    """best-improvement one-for-one swaps; a swap is taken only if gate(new, cur)."""
    cur, now = list(cur), scorer(list(cur))
    for _ in range(rounds):
        moves = [(i, j) for i in range(len(cur)) if cur[i] not in keep for j in pool if j not in cur]
        vals = batch(scorer, [cur[:i] + cur[i + 1:] + [j] for i, j in moves])
        took = False
        for v, (i, j) in sorted(zip(vals, moves), key=lambda x: -x[0])[:50]:
            if not v > now:
                break
            new = cur[:i] + cur[i + 1:] + [j]
            if gate is None or gate(new, cur):   # the best IMPROVING move that clears the gate
                cur, now, took = new, v, True
                break
        if not took:
            break
    return cur
```

`dota/env/som/frontier.py`:

```python
SWAP_T = 2.0

def select(world, cfg, sets=(), exclude=(), counts=None, ds=None):
    """greedy fill on val from empty and from each start in `sets`; swaps on every start;
    the incumbent (grown from the first given start = the last F) is replaced by another
    start's set only at the swap bar."""
    ds = ds or DayStats(world)
    f = scorer(world, cfg, ds, counts)            # IC from per-day moments when cfg.objective == "ic"
    pool = [j for j in range(len(ds.names)) if finite(j) and ds.names[j] not in exclude]
    starts = [()] + [tuple(ds.idx[n] for n in s)[:cfg.k_max] for s in sets if s]
    t = _paired_t(world, cfg, ds, ones)           # paired-by-day t on val
    got = [swap(greedy(ds, None, cfg, ones, pool, -np.inf, cfg.k_max, start=st, scorer=f,
                       stop_flat=True),
                pool, f, cfg.k_max, gate=lambda a, b: t(a, b) >= SWAP_T)
           for st in starts]
    best = max(got, key=f)
    if len(got) > 1 and best is not got[1] and t(best, got[1]) < SWAP_T:
        best = got[1]                             # path switches lost 0.0185 on test (LU S7a)
    return [ds.names[j] for j in best]
```

Selecting on IC while scoring on EV is one override in `dota/agent_plan.py`:

```python
def _parts(cfg, out):
    """the selection in `select_objective` when set: on LU, F chosen on ev_drift's val days
    scored 0.963-0.968 on test, F chosen on IC from the same banks 0.995-1.084."""
    ing, splits, legal, trainer, nc = _bank_parts(cfg, out)
    if cfg.get("select_objective"):
        nc = replace(nc, objective=cfg["select_objective"])
    return ing, splits, legal, trainer, nc
```

The world is still built with `objective="ev_drift"`, so it loads the 30s and 60s labels.
`Campaign.test()` and paired test gains read `cfg["objective"]`. Selection, the surrogate's
gains and every val number the agent sees read `nc.objective` (IC).

### 11.4 The surrogate (a baseline, and a tool the agent steers)

`dota/bank.py: learn` gives each trained model its marginal val gain to F. For a member, that
is what dropping it loses; for any other model, what adding it gains. A random forest is fitted
on those gains over the one-hot recipe axes.

`dota/env/som/surrogate.py`:

```python
class Surrogate:
    def fit(self, cs, gains):
        self.cols = sorted({(a, c[a]) for c in cs for a in self.axes if a in c})
        self.rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                        max_features=0.5, random_state=self.seed)
        self.rf.fit(self._x(cs), np.asarray(gains, float))
        return self

    def samples(self, cs):                       # (trees x recipes): the posterior draws
        return np.array([t.predict(self._x(cs)) for t in self.rf.estimators_])

    def thompson(self, cs, k, cost=None, rng=None):
        """k distinct recipes, each the argmax of one drawn tree."""
        rng = rng or np.random.default_rng(self.seed)
        draws = self.samples(cs)
        draws = draws + rng.uniform(0, 1e-12, draws.shape)   # break exact ties per seed
        picked = []
        for t in rng.permutation(len(draws)):
            order = [i for i in np.argsort(-draws[t]) if i not in picked]
            if order:
                picked.append(int(order[0]))
            if len(picked) == k:
                break
        return picked
```

### 11.5 The round loop

`dota/agent.py: run`, condensed (`reveal_or_train`, `all_picked` and `write_ledger` stand for a few inline lines each):

```python
for it in range(len(done), iters):
    C.rebuild(bank)
    names = C.select(prev)                       # F, from the last F
    F = C.node(names)
    sur, gain, cs, _ = learn(C.world, C.nc, names, C.legal, C.ing, seed=it)
    untrained = [n for n in C.universe if n not in gain and n in cs]       # benchmark mode
    brief(C, ws, it, F, diagnose(C.world, F.members), gain, ledger, budget, lib,
          prod_val=C.node(prod).score, available=untrained)
    listener = serve(C, ws + "/.tool.sock", F, sur, gain, cs, untrained)  # ./tool answers
    plan = plan_agent(C, ws, log)                # the LLM session -> JSON plan
    listener.close()

    picks, rows, left = {}, [], models
    for e in plan["experiments"][:4]:
        p, facts = realize(C, e, F, untrained, sur, cs, left, ws, rng)
        p = [r for r in p if r not in {x for q in picks.values() for x in q}]
        picks[e["id"]] = p
        left -= len(p)
    if left > 0:                                  # unspent budget: surrogate Thompson draws
        fill = [n for n in untrained if n not in all_picked]
        picks["_fill"] = [fill[j] for j in sur.thompson([cs[n] for n in fill], left, rng=rng)]

    bank = reveal_or_train(bank, all_picked)      # benchmark: append pre-fitted members
    C.rebuild(bank)
    new = C.select(names)
    for e in plan["experiments"]:                 # the agent's own ensemble, at the swap bar
        if e["action"] == "ensemble":
            g, se = C.paired(C.node(e["members"]).members, C.node(new).members, "val")
            if se and g / se >= 2.0:
                new = e["members"]
    test_gain = C.paired(C.node(new).members, F.members, "check")   # sealed
    write_ledger(rows, lessons=plan["lessons"])   # val-only fields
    prev = new

ep = experience.record(C, prev, out, cfg, ledger, iters_rows)   # online learning
if cfg.get("experience"):
    experience.add(cfg["experience"], ep)
```

`realize` turns `suggest` into Thompson draws inside the agent's region:

```python
if a == "suggest":
    pool_ = [u for u in untrained if matches(cs[u], exp.get("where") or {})]
    k = min(int(exp.get("n") or 8), left, len(pool_))
    return [pool_[j] for j in sur.thompson([cs[u] for u in pool_], k, rng=rng)], {"pool": len(pool_)}
```

### 11.6 Online learning: the experience store

`dota/experience.py`:

```python
def record(C, F_names, out, cfg, ledger, done):
    """out/episode.json: the finished campaign, val-only."""
    _, gain, cs, _ = learn(C.world, C.nc, F_names, C.legal, C.ing)
    ep = {"name": ..., "policy": ..., "objective": cfg["objective"],
          "trajectory": [{"models": cumulative_trained, "F_val": d["F_after_val"]} for d in done],
          "F_start_val": done[0]["F_val"], "F": F_names,
          "gains": {n: g for n, g in gain.items()},           # marginal val gain to the final F
          "coords": {n: cs[n] for n in gain if n in cs},
          "experiments": [val-only fields of each ledger row],
          "lessons": [every lesson the agent wrote]}
```

`render(store)` writes the two files the next campaign reads:

- **`experience.md`**, in this order:
  - one row per episode: its val trajectory by models trained, and its final F;
  - the best earlier final F on val;
  - the top 30 recipes by mean gain, with how many episodes trained each and how many final Fs held it;
  - a dead list: recipes trained in 2+ episodes, never in a final F, gain ≤ 0;
  - per-cell effects (the mean gain of each axis value and value pair, with SE and n);
  - the mean val gain of each kind of experiment, and how many of its models entered F;
  - the latest 20 lessons.
- **`recipes.json`**: every recipe the episodes trained, with its mean gain, episode count
  and how many final Fs held it, so the agent can grep it.

### 11.7 The brief (`dota/prompts/agent_v9.md`): the parts that carried the result

The brief is the agent's program. These paragraphs are what moved the results:

- **Memory (offline).** "`memory.md` holds what won on OTHER symbols … The surrogate starts every
  campaign knowing nothing; you can start from those priors. Use them to choose the first
  iterations' recipes (the anchor and its complements), then let this symbol's evidence take
  over." Without it, the agent trailed the surrogate on IC. With it, it led at 8-24 models.
- **Experience (online).** "Iteration 0: train the recipes that paid in more than one episode,
  and propose the best earlier final F (all of its members trained) as an `ensemble` once they
  are in the bank … one trained in several episodes that never entered a final F is dead: skip
  it. Then spend the budget where the earlier episodes were thin … Write lessons that a later
  campaign can act on."
- **Where the edge is (measured).** "Recipes the agent picked by hand did worse than the
  surrogate's Thompson draws at the same budget … Give most of each iteration's training budget
  to `suggest` … Train recipes by hand only for a hypothesis the surrogate cannot see." The
  other lever named there is the ensemble: run `./tool search` and propose sets at t ≥ 2 that also have a
  positive held-out gain.
- **Noise discipline.** "With hundreds of trained models, a val gain of +0.001 is usually selection
  noise … A new model has to clear [paired t ≥ 2] too."
- **Selection proxy.** When `state.json` has `val_numbers_in: ic`, the val numbers are IC and
  test is EV, and the brief says why.

### 11.8 What the agent actually does (episode 3, chain 1, round 0)

```
e1 train    11  "Rebuild the 11 untrained members of the best earlier final F (LU_EIc1e2,
                 val 1.0749). Each one paid in 2 episodes ..."                  -> 10 entered F
e2 train     2  "Cross-symbol prior: the usual second leg is a 10s (or 1s) tree on agg_store,
                 and no LU episode has trained tree x agg_store at 10s or 1s."  -> 1 entered F
e3 suggest   3  "tree x agg_noflow on target y is unmeasured on LU in both earlier episodes"
e4 ensemble     "The best earlier final F, all 12 members, once e1 lands."
lessons: "search with 34 models: the best set is t=1.97 over F, but held_out_gain_of_searching
          is -0.0049; I proposed no ensemble."
```

Round 0 spends most of the budget rebuilding what its own past found (the online part). It
puts a few models on gaps flagged by the cross-symbol memory (the offline part), and lets the
surrogate cover unmeasured cells. That is why it reaches ~1.05 at 16 models, where the
baselines are at ~0.99. It also shows the plateau mechanism: from round 1 on, F is full, and
nothing new clears the swap gate.

### 11.9 Known weaknesses of this reference

- Flat after 16 models (§10.1): the cap plus the gate freeze F. The rebuild of the best-val
  earlier F carries a winner's curse across episodes.
- Episodes are not monotone (1.069 → 1.062 from episode 3 to 4).
- `dota/agent.py` and `dota/agent_plan.py` are 321 lines each. The 400-line cap is respected,
  but the loop is not yet reviewed for merge to `futs`.

## 12. Where things are

| What | Where |
|---|---|
| Design | `docs/adr/0452-dota-agent-loop.md` |
| Code | branch `dota-agent-loop` (worktree `/qr_scratch/tony/wt/dota-loop`): `dota/agent.py`, `dota/agent_plan.py`, `dota/experience.py`, `dota/env/som/{frontier,select,tools,surrogate,objective,world}.py` |
| Briefs | `dota/prompts/agent_v8.md` (memory), `agent_v9.md` (memory + experience) |
| Universe | `/qr_scratch/tony/dota/agent/universe_LU_nor2f0.json` (643 pre-fitted) |
| Campaigns | `/qr_scratch/tony/dota/agent/LU_EI*` (one dir per run; `iters.jsonl`, `ledger.jsonl`, `episode.json`) |
| Experience stores | `/qr_scratch/tony/dota/agent/experience/LU_ei_c{0..5}` |
| Scoreboard | `/qr_scratch/tony/dota/agent/diag/score_ev.py`, `scoreboard_EI_20261023.txt` |

## Glossary

- **IC**: correlation of the prediction with the forward return, over all rows.
- **EV@h**: average signed forward return over horizon h on the top-1% predictions, in ticks.
- **Recipe / member / signal**: one trained small model, named by its axis values.
- **Bank**: all trained members of a campaign. **Universe**: all pre-fitted members (benchmark).
- **F**: the current ensemble. **k_max**: its size cap (12).
- **Greedy fill**: add the member that most raises val until no add helps or F is full.
- **Swap gate**: replace a member only when the paired val gain has t ≥ 2.
- **Thompson sampling**: pick the best recipe under one random draw of the surrogate, so
  uncertain regions get explored in proportion to their chance of being best.
- **Episode / chain**: one campaign; a sequence of campaigns sharing one experience store.
