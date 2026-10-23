# Case study: adsbench as a learning/optimization problem

A summary of adsbench in the shape shared by every problem a general self-improving optimization agent will face:
**data (with a train/test split), a goal, an evaluation function, and an agent that writes code to score well on
the held-out evaluation by learning from the training data.** It covers the core prediction problem, the three
optimization campaigns already run on it (the judge's accuracy, "AdsBrain" against frontier LLMs, and the
scaling curve), what worked, what didn't, and the traps a general agent has to avoid.

Sources: `README.md`, `docs/metric.md`, `docs/judge.md`, `docs/learning.md`, `docs/scripts.md`,
`docs/judge_bench.md`, `docs/data.md`, `docs/scaling_handoff.md`, and the transcripts of the "scaling"
(2026-10-22/23), "adsbrain" (2026-10-21/23) and archived "AdsBrain" (2026-10-16) sessions. Numbers are quoted as
reported. Different evaluation protocols give different numbers, and they are **not comparable** with each
other (see §5).

---

## 0. TL;DR

| | |
|---|---|
| **Task** | Given two ad creatives (copy, images, video stills, video script) that run in the same setup (advertiser, market, launch day, targeting, placements), predict which one Meta's delivery will give **more reach**, and by how much. |
| **Unit of data** | A *pair* of ads from one *cohort*: ads one advertiser launched the same day with the same targeting and placements. The label is **PRA = log10(reach_A / reach_B)**. |
| **Data** | Meta Ad Library (via ScrapeCreators), 5 EU/UK markets. The latest build has 219,259 ads in 37,843 cohorts (116,644 video, 102,615 image). The ad view has 1.52M comparisons. The script view has 101,417 script rows and 635,551 comparisons. Over 1,000 advertisers. |
| **Primary metric** | **Direction accuracy**: the weighted share of held-out pairs where the predicted winner is right (0.5 = chance). The benchmark form is the **AdsBench score = 100·(2·acc − 1)**. |
| **Splits** | Cohort-holdout cross-validation (within-advertiser memory), advertiser holdout (transfer), and a **forward-in-time stratified benchmark** (the honest one: ads that haven't run yet). |
| **Current best** | `nn:joint` (nearest neighbours over embedding differences of logged pairs): 0.581 cohort-holdout on the ad view, **0.577 accuracy / AdsBench 15.4** on the forward-only benchmark. Frontier LLMs score 4.9 to 9.2. |
| **Ceiling** | Label noise is huge: ads with an identical script in the same cohort differ by a median of 2.3× in reach. Only about 20% of reach-gap variance looks real, and the estimated attainable ceiling is about 0.70. |
| **Hardest open problem** | **Transfer across advertisers.** With the scored advertiser held out, almost every model is at chance (0.50 to 0.52). Nearly all of the signal is "this advertiser's similar creative did well before". |
| **Optimization campaigns run** | (1) Push held-out accuracy to 0.65: plateaued at about 0.62. (2) Make an LLM-based "AdsBrain" beat `nn:joint` out of sample: best +0.5 points, not significant. (3) Make the "own advertiser fixed, add other advertisers" learning curve reach a slope of +0.01 per 10× data: best honest +0.003. |

---

## 1. The problem

### 1.1 Why it exists

The product is an online call: **"will a planned video-ad script (or finished ad) reach more people than another
one, in this setup, and by how much?"** (`Server.compare_script`, `Server.compare_ads`). A creative team uses it
before shooting or launching. In the longer-term design (`docs/learning.md`) the same model is the **reward model
(the "Judge")** for an RL-style loop: an LLM creative generator and a testing policy learn against it, with
Dream-RSI-style replay of logged tests.

### 1.2 The label: reach, relative to a cohort

The Ad Library has **no spend, clicks or conversions**, only an estimated cumulative reach per ad (plus
targeting, placements and dates). adsbench turns that into a creative-quality signal through cohorts:

- A **cohort** is a set of ads that share the same advertiser page, start date, targeting (locations, ages,
  gender) and placements. They very likely share an ad set and its budget. Meta's delivery splits that budget
  among the creatives according to what it predicts will perform, so within a cohort reach mostly reflects the
  creative.
- **CRA** (Cohort Reach Advantage): `log10(R_i) − mean_j∈c log10(R_j)`. It works like an RL advantage, with the
  cohort mean as the baseline (GRPO-style group-relative).
- **PRA** (Pairwise Reach Advantage): `log10(R_A / R_B) = CRA_A − CRA_B`. The cohort mean cancels, so each pair
  is self-contained evidence. **PRA is the regression target and its sign is the classification target.**

### 1.3 Formal statement

```
input   x = (setup s, creative A, creative B)
          s: market ∈ {GB, DE, FR, NL, BE}, launch_date, age range, gender, placements, [advertiser]
          creative: primary text, headline, description, images (≤10), video stills, video script
          (the script view sees only s + the script; the ad view sees everything)
output  f(x) → (PRA_hat ∈ ℝ, p(A wins) ∈ [0,1], an 80% interval, support level)
label   y = PRA(A vs B) = log10(reach_A / reach_B), observed after the ads ran
goal    maximise direction accuracy  E[ 1{sign f(x) = sign y} ]  on pairs from unseen cohorts
        (and, for the benchmark, unseen *future* cohorts), with honest calibration
subject to  the online call staying fast (~0.6 s mean, CPU, no LLM by default)
```

The task is antisymmetric: f(A, B) = −f(B, A). The evaluation asks half the pairs loser-first, so a bias like
"always say A" can't raise a score.

---

## 2. The data

### 2.1 Source and collection

- **Meta Ad Library**, collected through the **ScrapeCreators** API: one credit per list call (about 10 to 30 ads
  each) and one credit per ad detail. Reach only comes with the detail call. Media (images, videos) are saved
  by content hash in `gs://adsbench-arrival-platform/media/`, and snapshots are stored as Parquet.
- **Advertisers** are tiered around the client (Joybuy): T0 Joybuy, T1 marketplaces (Temu, Amazon, AliExpress,
  SHEIN, …), T2 retailers in Joybuy's categories, T3 brands and the long tail (`config/advertisers.json`).
- **Video scripts** are extracted from each video by an LLM (Gemini via OpenRouter; about $650 for about 139k
  videos), parsed into a screenplay format (scene headings, action, VO, dialogue, SUPER on-screen text), and
  optionally tagged by an LLM (structure, hook, humour, presenter, CTA, music, …).
- **Embeddings:** OpenCLIP ViT-B-32 with a multilingual XLM-R text tower. Each ad becomes `[text | image | script]`,
  three 512-d halves, each L2-normalised, plus per-element vectors (body, headline, description, images, video,
  and four script elements).

### 2.2 Size over time (the data grew about 125× in two weeks)

| Build | Ads | Cohorts | Comparisons | Advertisers |
|---|---|---|---|---|
| 2026-10-05 (Joybuy UK) | 100 (62 in cohorts) | 18 | 127 | 1 |
| 2026-10-08 export | 1,412 | 210 | 12,298 | 9 pages |
| 2026-10-10 unified | 1,743 | 242 | 16,410 | 15 pages |
| 2026-10-15 views | 10,433 | 880 | 176,554 (ad view) | – |
| 2026-10-16 video scale | 15,664 | 2,228 | 48,179 (script view) | 63 / 80 |
| 2026-10-18 100k scripts | 122,518 | 19,537 | 635,551 (script view) | 257 |
| **2026-10-20 (current)** | **219,259** | **37,843** | **1,524,275 ad view / 635,551 script view** | 1,065 (the scaling session's count) |

The current build spans launches from 2025-10-08 to 2026-10-08 (the Ad Library drops an EU ad a year after it
last ran).

### 2.3 Two "views" (what is known when the decision is made)

| View | Rows | Features | Answers |
|---|---|---|---|
| `ad` | one per ad | setup + `[text \| image \| script]` | finished-ad comparisons (`compare_ads`, the benchmark) |
| `script` | one per **distinct script in a cohort** (ads with the same script, at cosine ≥ 0.98, are merged; their outcome is the mean log reach) | setup + script only | pre-production comparisons (`compare_script`) |

A script query is a **written script with no footage**, but the training rows are scripts *extracted* from
videos. That is a train/serve distribution shift. The style-shift test (§4.6) found the verdict agrees 82% of
the time after rewriting the scripts in a pre-production style, and accuracy drops from 0.572 to 0.551.

### 2.4 Filters applied before cohorts are formed

- **No dynamic-creative (DCO) ads.** Their reach is shared by several creatives.
- **Minimum reach:** 1,000 for videos, 100 for images. Below that the ad mostly recorded being stopped. (For
  GB only UK reach is reported, so far fewer GB image ads clear 1,000.)
- **One market per ad.** An EU ad's reach is EU-wide, so copies in other markets add nothing.
- Cohorts of fewer than 2 ads give no pairs.
- **Pair weight** is 1/(n−1) inside an n-ad cohort, so a cohort of n ads (n(n−1)/2 pairs) counts about n units,
  not n².

### 2.5 Dataset skew (it matters for every split)

- About 83% of ads target ages 18 to 65+, 81% all genders, and 59% use automatic placements. Meta's delivery,
  not the setup fields, picks the real audience, so **setup features carry almost no signal** (every setup
  ablation tied with ignoring the setup).
- Advertisers are heavy-tailed. Temu alone is a big share, and random cohort samples lean towards the big
  advertisers, where accuracy is higher (0.644 on advertisers with 10+ benchmark pairs, against 0.576 on the
  rest).
- Advertisers relaunch near-identical creatives a lot. **85% of scripts have their nearest neighbour inside
  their own advertiser, and 71% have a near-copy (similarity ≥ 0.95) in their own history**, against 16% among
  other advertisers.

---

## 3. Labels, noise and the attainable ceiling

This is the defining difficulty of the problem.

- **The noise floor:** ads with identical copy in one cohort differ by about 1.44× in reach (the early build's
  measure). In the current build, ads with an identical *script* in one cohort differ by a median of
  **2.3×** (|PRA| = 0.364). The build computes these floors per view (`server/confidence.py`).
- In the archived AdsBrain research, only about 20% of reach-gap variance was estimated as real, giving an
  estimated ceiling of about 0.70 direction accuracy.
- **"Clear" pairs.** The benchmark keeps only pairs with |PRA| ≥ 0.364. Cleaning the *training evidence* the
  same way raised cross-validated accuracy from 0.578 to 0.605. A 10× cut-off reaches 0.648, but it keeps only
  19% of pairs, so it was not adopted.
- **Label validity caveats:**
  - Reach is Meta's delivery decision, not sales or cost per result.
  - Reach is an estimate whose update cadence is unknown.
  - Catalogue and multi-version ads show different things to different viewers.
  - Cohorts are inferred from public fields, so two campaigns can be merged by accident.
  - An ad stopped early has less reach, which is partly a performance signal and partly calendar time.
- **pCRA@N** (the projected advantage from the first N days of reach) is designed but blocked on daily
  snapshots. It would turn the label from "weeks later" into "days later", which is needed for any live-test
  loop.

**Consequence for an optimization agent:** effect sizes are a few accuracy points, and confidence intervals on
realistic test sets are ±1.5 to ±4 points. Most "improvements" an agent finds on a few hundred pairs are noise.
The harness must enforce paired tests and a fixed test set used once.

---

## 4. Evaluation functions

### 4.1 Direction accuracy (primary)

The weighted share of held-out comparisons where sign(prediction) = sign(PRA). A prediction of exactly 0 gets
half credit. Weights keep each cohort's total weight fixed. The `clear` variant uses only |PRA| > 0.15 (or
0.364 in the benchmark).

### 4.2 AdsBench score (benchmark form)

`100 × (2 × accuracy − 1)`: 0 is a coin flip, 100 is always right, and it can go negative. It is
stratum-weighted (each stratum by its share of real cohorts). Intervals come from a cohort bootstrap within
strata, and the paired comparison against a reference uses McNemar's exact test. Also reported:

- accuracy of the most confident 10%, 20%, …;
- results by stratum, market, format and gap size;
- the Brier score of P(A);
- position bias (how often A is picked);
- agreement between systems;
- cost and latency per pair.

### 4.3 Chance baseline (null)

The whole evaluation is rerun with reach shuffled among the ads of each cohort (3 to 10 runs). Holding data out
can push null scores below 0.5, so **a model counts only if it beats the null's 95th percentile**, not 0.5. On
small data that bar is high: with 18 cohorts it is about 0.62.

### 4.4 Secondary metrics

- **Pick / PPV:** the mean CRA of the ad ranked first in each held-out cohort. Random = 0; the best possible is
  about +0.45.
- **Spearman / RMSE** of predicted against actual PRA.
- **Calibration** (`calibration.json`): a held-out fit maps raw scores to an expected PRA, an 80% interval and
  P(win). The checks are 80% interval coverage (0.81 to 0.84 achieved) and the slope of actual on calibrated
  (0.91 to 0.99). The intervals are honest but wide: about ±1.1 log10, roughly 0.03× to 30×.
- **Reliability by support:** accuracy by quartile of evidence weight. For the ad view it is 0.532 / 0.588 /
  0.588 / 0.614 from low to very high support. Near-copy pairs (top similarity ≥ 0.85) score about 0.64 to 0.73.
  The online answer reports this per call, so the answer never claims more than the evaluation shows.
- **Learning curve:** accuracy against log10(training data), with the slope per 10× and a 90% bootstrap interval.
  This became a target in its own right (§7.3).

### 4.5 Hard constraints (not scored, but binding)

- **Latency:** `compare_script` takes a mean of 0.6 s and p90 of 0.75 s on CPU, with no LLM call by default.
  Optional work (LLM tags) is opt-in. Any change on the call's path is benchmarked
  (`tests/bench_predictor.py`).
- **Honesty:** every estimate carries its support and held-out accuracy. No field may claim more than the
  evaluation shows.
- **Reproducibility:** builds are sealed versions (`warm/<day>-<commit>/`, an md5 manifest), and they are never
  edited. A re-evaluation is a new version.
- **Budget:** collection is limited by ScrapeCreators credits, and LLM experiments by spend caps
  ($300 to $400 per campaign).

### 4.6 Style-shift test (script view)

Held-out scripts are rewritten by an LLM in a pre-production register and judged again. This measures the
train/serve gap described in §2.3.

---

## 5. Train/test splits, and why the choice of split is the whole game

There are four split protocols in use. They answer different questions, and **their numbers are not
interchangeable**.

| Protocol | How | What it measures | `nn:joint` (current build) |
|---|---|---|---|
| **Cohort holdout CV** (build default) | 10 folds, or 5 at scale, over cohorts. The held-out cohort's pairs are never in the evidence, but its advertiser's *other* cohorts are, including later ones. At scale: 5,000 fixed held-out cohorts and 12 ads per large cohort. | Within-advertiser memory, with future leakage | ad view 0.581, script view 0.570 |
| **Advertiser holdout** | One fold per advertiser. The scored advertiser is never in the evidence. | Transfer to a new advertiser | about 0.50 to 0.51 (chance) |
| **Forward-only benchmark** (`ad-2026-08-09-60d`) | Test cohorts launched 2026-08-09 to 10-08. The model may search only cohorts launched before 2026-08-09. Stratified into 5 markets × {both video, both image}, 360 test + 40 dev pairs each (3,600 + 400). Clear pairs only. At most 10% of a stratum from one advertiser. A coin flip decides which ad is A. | Ads that haven't run yet: the product's real situation | **0.577, AdsBench 15.4 (12.5–18.2)** |
| **Training windows for hill-climbing** | `ad-2026-06-10-60d-train` (2,000 pairs; evidence before 06-10; `nn:joint` scores 11.0, unlike test). `ad-2026-07-10-30d-train` (1,987 pairs; evidence before 07-10; `nn:joint` 15.1, matching test's 15.3). | Somewhere to select methods without touching test | – |

The scaling session also built three fixed **exams** on one store (the old 1,360 cohorts, 1,500 random cohorts,
and a late window). Only an *allow-mask* over the evidence changed between conditions, so the data seen was the
only variable.

### 5.1 Leakage and selection traps that were actually hit

1. **Future relaunches.** Cohort holdout lets the model see *later* relaunches of the same creative. That gives
   0.598 against 0.563 forward-only on the first benchmark. 12.3% of pairs had a near-copy under cohort holdout
   against 4.2% forward-only, and near-copies are right about 73% of the time against 56–58% without.
   **Forward-only is the honest number.**
2. **Dev/test cohort leak across strata** in the first version of benchmark v2. Found and fixed.
3. **Judges that fit on the whole store** (`linear`, `nn:separate`'s fallback) see the future in a forward-only
   protocol. They were excluded from the ensemble search.
4. **A reshuffled split.** Cleaning the evidence reshuffled the held-out cohorts, which produced spurious ±0.015
   "effects". Fixed by pinning the split.
5. **Winner-first storage.** Evidence stored winner-first gave uninformative pairs a vote of "A wins", which the
   evaluation rewarded. Fixed with zero weight for uninformative pairs and by asking half the pairs loser-first.
6. **Best-of-N on dev.** Each LLM's best-of-3 prompt on dev scored 2–6 points lower on test. A lucky seed (0.622)
   averaged 0.618 over three seeds.
7. **Training windows unlike test.** `nn:joint` is weaker on the early window (11.0 against 15.3), which made
   LLM overrides look good twice. They didn't hold on test. The fix was a matched window (§5, last row).
8. **Reused exams.** The scaling loop reused the same 1,500 test cohorts across about 8 rounds, so its best
   result (0.570) needs a fresh confirmation.
9. **Learning curves from thinning.** Random subsampling of one build removes each held-out cohort's nearest
   evidence (own-advertiser near-copies). The resulting curve predicts gains that real growth, which adds *new*
   advertisers, never delivers (§7.3).

---

## 6. Baselines and current results

### 6.1 Models in the codebase

| Model | Idea |
|---|---|
| `zero` | Always 0, so accuracy is exactly 0.5 |
| `nn:joint` (**default**, "adsbrain") | For a query change d = A − B, find logged pairs whose embedding difference is close (cosine kernel, τ), weight them by setup similarity and pair weight, and average their signed margins with shrinkage w0. Tuned to take 300 neighbours from 3,000 IVF-PQ candidates, with another advertiser's evidence ×0.1. |
| `nn:separate` | The same, matching element by element (body, headline, description, images), with penalties for confounded elements. Interpretable (which part drove the verdict). |
| `nn:text`, `nn:image`, `nn:joint:noscript` | Ablations on a subset of the vector |
| `linear`, `linear:tags` | Ridge regression on embedding differences, or on hand-made and LLM tags |

### 6.2 Headline numbers

**Cohort holdout, current build `2026-10-20-1a88af9609f4`:**

| View | `nn:joint` | Ablation | Chance p95 | Learning curve (5 / 15 / 40 / 100%) | Slope per 10× |
|---|---|---|---|---|---|
| ad | 0.581 | `noscript` 0.583 | 0.503 | 0.532 / 0.544 / 0.564 / 0.581 | +0.039 |
| script | 0.570 | `linear:tags` 0.505 | 0.506 | 0.531 / 0.535 / 0.549 / 0.570 | +0.030 |

**Forward-only benchmark, 3,598 pairs, 442 advertisers:**

| System | AdsBench (90% CI) | Accuracy | Top-10% score | vs `nn:joint` | $ / 1k pairs |
|---|---|---|---|---|---|
| **`nn:joint` (adsbrain)** | **15.4 (12.5–18.2)** | 0.577 | 52 (76% right) | – | ≈ $0 |
| Gemini 4 Argon, v1 prompt | 9.2 (6.3–12.2) | 0.546 | 14.1 | −6.1 (p 0.03) | $19.89 |
| Claude Opus 5.5, v2 prompt | 5.0 (2.1–8.2) | 0.525 | 4.2 | −10.3 (p < 0.001) | $36.69 |
| GPT-6.1 Sol, v1 prompt | 4.9 (1.9–7.8) | 0.525 | 15.6 | −10.5 (p < 0.001) | $8.46 |
| LLM AdsBrain "follow1" (Gemini 3.8 Flash over `nn:joint`'s case file) | 15.8 | – | – | +0.5 (CI −2.2 to +2.9, p 0.75) | – |

**Advertiser holdout (transfer):** every embedding model is at chance. Only hand-made, language-independent
tags showed a trace of transfer on the 1,412-ad build (0.526, just above the null's p95).

---

## 7. The optimization campaigns already run (what an agent would be doing)

Three distinct goals have been posed to agents on this problem. Each is a ready-made task for a general
optimizer.

### 7.1 Campaign A: raise held-out accuracy to 0.65 (archived "AdsBrain" session)

- **Setup:** cohort-holdout CV with a fixed split, on the ad and script views. Phase-2 baseline 0.605 after
  noise-floor cleaning of the evidence.
- **Tried (runs m1 to m19):**

| Approach | Accuracy |
|---|---|
| Direction-only votes | 0.610 |
| + within-cohort consistent scores (Bradley–Terry) + setup weights | 0.6165 |
| Cross-cohort pairs (3 seeds) | 0.6218 / 0.6185 / 0.6126 (mean 0.618: "m16") |
| Ridge, pairwise neural networks | 0.54–0.55 |
| Stacking | 0.605–0.609 |
| Learned similarity | 0.584 |
| LLM judge (Gemini 3.8 Flash) | 0.502 |
| Best ad-view variant | 0.6196 |

- **Outcome:** a plateau at about 0.62. The 0.65 goal was not met. The conclusion: the noise ceiling is about
  0.70, more data gives about +0.03 per 10×, and parametric models lose to nearest neighbours because the
  signal is mostly memory of near-identical creatives.
- **Earlier tuning, in the build** (`docs/judge.md`): the candidate pool, not the neighbour count, was the limit
  (25 of 1,000 gave 0.563, 300 of 3,000 gave 0.574, and with the advertiser weight 0.581). Nothing else moved it
  beyond noise:
  - aggregation, temperature, market and season weights;
  - exact indexes;
  - other text embedders (e5, bge-m3, gte);
  - ensembles;
  - ridge 0.522, MLP 0.529;
  - Bradley–Terry 0.545 to 0.567;
  - a fine-tuned encoder (collapsed).

### 7.2 Campaign B: make an LLM-based "AdsBrain" beat `nn:joint` out of sample (the "adsbrain" session)

- **Step 1, benchmark first.** Build the forward-only stratified set (§5), with a dev split for prompt
  selection, and run three frontier LLMs with three prompts. Result: the LLMs score 4.9 to 9.2 against
  `nn:joint`'s 15.4. They agree with each other 71–78% and with `nn:joint` 45–46%: a shared idea of a "good ad"
  that only weakly tracks delivery. Their stated P(A) is no better than always saying 0.5 (Brier 0.256 to
  0.275). Gemini picks A 64–65% of the time. Adding the advertiser's own history to the prompt (v3) helped no
  model. Cost: $82 for the prompt study and $234 for the test.
- **Step 2, hill-climb an LLM final stage.** It reads a *case file*: `nn:joint`'s verdict, the retrieved
  evidence, explorer lessons and the advertiser's history. Budget $300, raised to $400.
  - Ads only → full case file on 600 training pairs: Gemini 3.8 Flash 8.0 → 13.0, Claude Haiku 5.5
    −0.5 → 11.6, DeepSeek v4.1 Flash 3.3 → 8.8, GPT-6 Luna 2.0 → 8.5. `nn:joint` scored 13.1 on the same pairs.
  - Ablations: removing images cost 5.1, history 2.7, evidence 2.4. Every variant landed between 9 and 13.
  - Free-hand finalist: +3.6 on training, then **test 13.2 against 15.3** (−2.1). It overrode `nn:joint` on 31%
    of pairs and was right 48% of the time when it did.
  - **"follow1"** (keep `nn:joint`'s pick unless there's a reason): +5.0 on training (p 0.02), +5.6 on its held-out
    part, **test 15.8 against 15.3 (+0.5, not significant)**. It overrode 17% of the time and was right 51% when
    it did.
  - Also tried, with no gain:
    - LLM attribute tagging of 47,575 ads plus a learned model (AdsBench 4.1);
    - single LLM ratings such as "appeal" (50.4%) and "visual quality" (48.1%);
    - independent frontier-LLM votes added to `nn:joint` (lowered it in every fold);
    - an LLM relevance re-scorer for the retrieved evidence;
    - a time-decay and advertiser-weight grid;
    - a sibling-judge ensemble (unfinished).
- **Outcome:** the LLM adds no independent signal on the hard pairs, so the best it can do is defer. Spend was
  about $252 by the end.

### 7.3 Campaign C: scaling, making learning from *other* advertisers work (the "scaling" session)

- **Observation:** each build's own learning curve (from thinning) looked log-linear and steep, but each bigger
  build landed **below** the previous curve's prediction. The 1.7k curve predicted 0.664 at 219k, and the build
  scored 0.581.
- **Diagnosis:** a fit over 33 curve points gives
  `acc ≈ 0.521 + 0.038·log10(share kept) + 0.010·log10(build size)` (rmse 0.0063). Thinning costs about
  +0.038 per 10×, but real growth (new advertisers, markets, months) only adds about +0.010 per 10×. On the
  fixed old exam, adding 13.6× more new data moved accuracy from 0.5789 to 0.5790. 82% of the model's weight
  came from the own advertiser.
- **Accuracy against the number of the advertiser's own cohorts in training:**

| Own-advertiser cohorts | 0 | 1–2 | 3–9 | 10–29 | 30–99 | 100–299 | 300+ |
|---|---|---|---|---|---|---|---|
| Accuracy | 0.503 | 0.518 | 0.520 | 0.536 | 0.547 | 0.578 | 0.547 |

- **Three curves defined on one 1,500-cohort exam:**

| Curve | 5% / 15% / 40% / all | Slope per 10× |
|---|---|---|
| 1. random thinning | 0.519 / 0.524 / 0.533 / 0.552 | +0.024 |
| 2. advertiser holdout | 0.499 / 0.502 / 0.500 / 0.510 | +0.007 |
| 3. own advertiser kept whole, others added | 0.556 / 0.556 / 0.554 / 0.552 | −0.004 |

- **The goal (set by the user):** make curve 3's slope +0.01 per 10×, with a guard (added by the agent) that the
  level must stay ≥ 0.552. This is an auto-research loop.
- **Rounds:**

| Round | Approach | Slope per 10× | Level |
|---|---|---|---|
| 1 | Own advertiser first in the candidate pool | +0.0022 | 0.551 → 0.554 |
| 2 | Advertisers weighted equally | +0.0097, **rejected** | starts at 0.509 |
| 2 | Shared + per-advertiser Bradley–Terry | +0.0045 to +0.0066 | 0.525 to 0.536 |
| 4 | TF-IDF text Bradley–Terry over everyone | −0.017 | 0.559 → 0.538 |
| 5 | Others weighted by closeness (same brand 1, same category 0.3, rest 0.1) | **+0.0032** (CI +0.0004 to +0.0059) | 0.551 → 0.555 |
| 6 | Text model per advertiser, own data only | 0 | **0.568** |
| 6/8 | Two-stage: shared text model, then a per-advertiser correction | +0.0009 to +0.0014 | **0.566 → 0.570** |
| 8 | Ceiling check: the shared text model on unseen advertisers | flat | 0.509 to 0.516 |

  Round 2's equal-weight result is the clearest gaming example: it raised the slope by lowering the start.
- **Outcome:** the goal was not reached. Scripts carry only about 1–1.5 points of signal that transfers between
  advertisers, and it is used up by 15% of the other advertisers. A by-product: a **two-stage per-advertiser
  text model scores 0.570 against 0.551** for `nn:joint` on that exam. It still needs a fresh test.
- **Why lessons don't travel:**
  - No trait wins in general: price 50.8%, voiceover 50.4%, on-screen text 49.0%.
  - Per-advertiser word models agree across advertisers at a median correlation of 0.02, against 0.21 between
    two halves of one advertiser's data.
  - Restricting to the *same exact setup* doesn't help: other advertisers' evidence then scores 0.499.
- **Open:** whether *visual* features transfer. The advertiser-holdout ceiling on the ad view was proposed but
  not run.

### 7.4 Campaign 0: data acquisition under a budget (the scaling handoff)

Not a learning problem, but an optimization one that feeds all of the above. The task is to spend ScrapeCreators
credits (20k, then about 240k) and LLM dollars to maximise *usable* pairs. Usable means in cohorts of 2 or more,
above the reach floor, not DCO, and not duplicated across markets.

Lessons:
- Listing cost 3× its estimate.
- Probe counts overstated yield 2–3×.
- 43% of image details reached fewer than 100 people.
- A list's run length and the page's history predict yield. `--min-days 2` keeps 98% of the useful ads.
- Parallel market plans fetched EU ads twice (26.6k credits wasted).
- The first runs hit a list of operational failures: rate limits, OOMs, credit exhaustion, hung workers, and
  quota limits.

---

## 8. What an agent can change (the search space)

| Layer | Levers | Notes |
|---|---|---|
| Data selection | Reach floor, DCO handling, noise-floor cleaning of evidence, cohort definition (relaxed cohorts, cross-cohort pairs), pair weighting | Cleaning the evidence was the single biggest gain (+0.027). Relaxed cohorts are still untested. |
| Representation | Embedder (CLIP multilingual, e5, bge-m3, …), script chunking, element vectors, tags (rule-based, LLM), view choice | Other embedders tied. Tags are weak but are the only thing that hinted at transfer. |
| Retrieval / model | Neighbour count, candidate pool, τ, shrinkage w0, setup weights, advertiser weight, time decay, own-advertiser-first, Bradley–Terry, per-advertiser models, two-stage shared + local, ensembles, LLM re-ranking or final stage | The candidate pool and advertiser weight mattered. Parametric global models lose. Per-advertiser local models look promising. |
| Calibration and answer | Support bins, interval fit, deferral or abstention | Honest uncertainty is part of the product. |
| Collection policy | Which advertisers, markets and windows to buy | The data that is collected decides which curve you are on. |

**Constraints on the search:** the online path must stay at about 0.6 s on CPU. Changes to `core`, `pipeline`
and `predictor` need human review. Builds are immutable versions. Experiments must be reproducible from a
sealed build.

---

## 9. Challenges (consolidated)

1. **Extreme label noise.** A 2.3× median reach gap between identical creatives, and about 80% of the variance
   is noise. Gains are a few points, and test CIs are ±1.5 points at best (3,600 pairs) and ±4 points on 600.
2. **Weak, local signal.** The predictive signal is mostly *memory*: this advertiser's near-identical creative
   did well before. It barely generalises across advertisers (chance under advertiser holdout), and creative
   "rules" flip sign between advertisers.
3. **Non-stationarity and time.** Relaunches make random CV leak the future. Forward-only scoring costs about
   3.5 points. Training windows differ in difficulty (11.0 against 15.3 on the same model).
4. **The split decides the answer.** Cohort holdout, advertiser holdout and forward-only give 0.58, 0.50 and
   0.577 for the same model. Learning curves from thinning mislead about the real scaling.
5. **Heavy-tailed advertisers.** Big advertisers dominate random samples and are easier, so stratification and
   per-advertiser caps change headline numbers by several points.
6. **Selection bias in iterative search.** Best-of-N prompts or seeds, a reused exam, and dev sets of 400 pairs
   all inflate results. A gameable metric (slope) needs a level floor.
7. **A label proxy.** Reach is Meta's delivery decision, not business value. LLMs' "taste" doesn't track it, so
   priors from world knowledge don't help.
8. **Multimodal, multilingual inputs.** Copy in 5 languages, images, video stills, and LLM-extracted scripts,
   with a train/serve shift between extracted and written scripts.
9. **Scale and compute.** 1.5M comparisons mean evaluation needs ANN indexes, sampling (5,000 cohorts, 12 ads
   each) and parallel folds. Naive scans took hours or ran out of memory. VM quotas (32 vCPU) and a 15 GB dev
   container limit the experiments. Spot VMs get pre-empted.
10. **Cost.** LLM baselines cost $8 to $37 per 1k pairs. Collection is limited by credits. An LLM-heavy approach
    must justify its cost against a $0 nearest-neighbour model.
11. **Latency.** Whatever wins must still serve in under a second without an LLM call, or be opt-in.

---

## 10. Packaging adsbench for a general self-improving optimization agent

### 10.1 Task card

```yaml
name: adsbench-pairwise-reach
type: pairwise ranking / binary classification with a real-valued margin
train_data:   sealed build warm/2026-10-20-1a88af9609f4 (ad + script views), restricted to cohorts launched
              before the cutoff (2026-08-09) and to evidence available before it
dev_data:     400 dev pairs of bench set ad-2026-08-09-60d (prompt and model selection only), plus
              ad-2026-07-10-30d-train (1,987 pairs, a matched forward window) for hill-climbing
test_data:    3,600 stratified test pairs of ad-2026-08-09-60d (scored once per finalist)
metric:       AdsBench = 100·(2·acc − 1), stratum-weighted; report a 90% cohort-bootstrap CI and a paired
              McNemar p against nn:joint
reference:    nn:joint = 15.4 (accuracy 0.577); best LLM 9.2; best LLM-over-nn 15.8 (n.s.)
secondary:    transfer (advertiser holdout), learning-curve slope (own advertiser fixed, others added),
              calibration coverage, top-10% accuracy
constraints:  forward-only (no evidence launched after the cutoff, no model fit on the whole store);
              online inference ≤ ~1 s on CPU without an LLM by default; LLM spend cap; no edits to sealed builds
```

### 10.2 What the harness must enforce (lessons from these campaigns)

- **A frozen test set the agent can't read**, scored once per finalist. A matched dev window for search.
  Report test only with a paired comparison against the reference.
- **Forward-only data access**, enforced by the harness (an allow-mask over the evidence by launch date), not
  by convention.
- **The chance baseline** (outcomes shuffled within each cohort) computed automatically. A result counts only
  above its 95th percentile.
- **Seeds and repeats:** report the mean over seeds, never the best seed.
- **Anti-gaming guards** on derived metrics: a level floor when optimizing a slope, a coverage requirement when
  optimizing accuracy on a subset (cleaning or abstention can raise accuracy by dropping hard pairs).
- **Track the budget per experiment** (LLM $, credits, VM hours) and fail fast on 402/429 errors.
- **An experiment log with results stored outside the container.** The scaling session's results were in a
  git-ignored folder lost with the container.

### 10.3 Promising directions not yet exhausted

1. **Per-advertiser / two-stage models** (0.570 against 0.551 on the scaling exam): confirm on fresh cohorts and
   on the forward benchmark.
2. **Visual transfer:** the advertiser-holdout ceiling on the ad view (image embeddings, 219k ads). Untested.
3. **Relaxed cohorts / cross-advertiser pairs** with a delivery-relative target (reach relative to the
   advertiser's week).
4. **Abstention as a product feature:** the top-10% most confident picks are right 76% of the time. Optimize
   coverage against accuracy rather than raw accuracy.
5. **A less noisy label:** pCRA@N and CRA@H from daily snapshots, and own-account live tests with spend data
   (cost per result).
6. **Collection policy as part of the optimization:** buy depth in the advertisers the product serves rather than
   breadth, given that breadth adds about +0.01 per 10×.

---

## 11. Pointers

| What | Where |
|---|---|
| Metric definitions | `docs/metric.md` |
| Judges, CV, chance baseline, tuning history | `docs/judge.md`, `source/core/judges/`, `source/pipeline/evaluate.py` |
| Views, script pipeline, calibration, results by build | `docs/scripts.md`, `source/core/views.py`, `source/pipeline/calibration.py` |
| Forward-only benchmark | `docs/judge_bench.md`, `source/pipeline/judge_bench.py`, `config/judge_bench.json`, `.claude/skills/run-ads-bench/` |
| Data, collection, filters | `docs/data.md`, `docs/scaling_handoff.md` |
| Learning-loop design (Judge as reward model, Dream-RSI) | `docs/learning.md` |
| Scaling research | branch `claude/scaling-curves-research`, `adhoc/scaling_curves_2026-10-22/`; artifacts "Why the Scaling Curves Disagree" and "Why Lessons Don't Travel" |
| LLM AdsBrain research | branch `claude/adsbrain-llm` (unmerged): `source/predictor/adsbrain.py`, `source/pipeline/adsbrain_bench.py`; bucket `bench/adsbrain/` |
| Bucket | `gs://adsbench-arrival-platform` (read through `source/core/storage.py`) |
