# Automated alpha research agents (LLM, RL, generative) and the statistics of backtest overfitting

Scope note: current as of 2026-10-23. Every system below is dated by its first arXiv or venue appearance. "Unverified" means I saw the claim only in a search snippet, an abstract or a secondary summary, not in the paper body. Sources I cite by DOI but did not re-open this session are marked "[not re-fetched]". Numbers are the authors' own and come from their own backtests unless stated otherwise. No system here has been independently replicated on a common benchmark.

**Naming collision (flag for the user's "AlphaAgent").**
- **AlphaAgent (Tang et al.).** "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay", by Ziyi Tang, Zechuan Chen, Jiarui Yang, Jiayao Mai, Yongsen Zheng, Keze Wang, Jinrui Chen and Liang Lin (Sun Yat-sen U., UNSW, NTU, CUHK-Shenzhen, Peng Cheng Lab).
  - arXiv:2502.16789: v1 24 Feb 2025, v2 9 Jun 2025.
  - Published at KDD '25 (Toronto, 3–7 Aug 2025), DOI 10.1145/3711896.3736838.
  - Code: github.com/RndmVariableQ/AlphaAgent.
  - Sources: [arXiv abs](https://arxiv.org/abs/2502.16789), [arXiv HTML v2](https://arxiv.org/html/2502.16789v2), [GitHub](https://github.com/RndmVariableQ/AlphaAgent).
- **AlphaAgentEvo.** A follow-up, ICLR 2026, "Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning". The mlanthology slug is `tang2026iclr-alphaagentevo`, which suggests the same first author. Sources: [ICLR](https://iclr.cc/virtual/2026/poster/10007685), [mlanthology](https://mlanthology.org/iclr/2026/tang2026iclr-alphaagentevo/).
- **The GitHub repo has changed.** The README I fetched now describes a China A-share "FactorZoo" framework: a Tushare panel of CSI 1000 constituents, 2015-01 to 2026-06, with a data package dated 2026-07-03. It does not mention the KDD paper or the Qlib/Baostock setup the paper uses. The repo appears to have been rewritten since publication ([GitHub](https://github.com/RndmVariableQ/AlphaAgent)).
- **"AlphaEvolve" is also taken twice.** An alpha-mining "AlphaEvolve" (computation-graph alphas, SIGMOD 2021, cited as [3] in AlphaGen) predates Google DeepMind's 2025 AlphaEvolve ([AlphaGen PDF, related work](https://arxiv.org/pdf/2306.12964)).

---

## Q1. Mechanisms and results of the main systems (AlphaAgent, RD-Agent(Q), Alpha-GPT, AlphaForge, AlphaGen, QuantAgent, FAMA, Chain-of-Alpha, AlphaSAGE, Alpha-R1, 2026 systems)

### Takeaway
Three designs dominate.
- **Generate-then-combine** (AlphaGen, AlphaForge, AlphaSAGE). An RL, GFlowNet or neural generator proposes formulas, and a linear "mega-alpha" combiner scores them by marginal contribution.
- **LLM research loops** (AlphaAgent, RD-Agent(Q), Alpha Jungle, Chain-of-Alpha, CogAlpha, AlphaMemo). A hypothesis leads to code, then a backtest, then feedback to the next step, with novelty, complexity and dedup filters.
- **LLM gating over a fixed pool** (Alpha-R1).

The pieces that transfer best to the DOTA ensemble search are these:
- AlphaGen's reward equals the marginal change in the combination's score.
- AlphaForge re-selects the top N by recent IC each period and fits OLS weights.
- RD-Agent(Q) schedules exploration with a contextual Thompson-sampling bandit, and in the paper's ablation this beat letting the LLM choose.
- AlphaAgent penalizes originality and complexity explicitly.

### Cited Findings

**AlphaAgent (Tang et al., KDD 2025; arXiv Feb 2025)**
- **Objective (Eq. 2).** It maximizes predictive performance L minus λ·R_g(f,h), where f is the factor and h the market hypothesis. — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
  - R_g = α₁·SL(f) + α₂·PC(f) + α₃·ER(f,h), with SL = symbolic length and PC = number of free parameters.
  - ER(f,h) = β₁·S(f) + β₂·C(h,d,f) + β₃·log(1+|F_f|).
  - S(f) is the maximum AST similarity to any factor in the "alpha zoo": the size of the largest isomorphic common subtree, normalized to [0,1].
  - C = α·c₁(h,d) + (1−α)·c₂(d,f), with α = 0.5. These are LLM-judged consistency scores in [0,1], for hypothesis vs. description and description vs. expression.
  - |F_f| is the number of raw features used.
  - The α, β and λ weights and the contents of the alpha zoo are **not reported**, and the sign convention of C inside ER is ambiguous (alignment should be rewarded, not penalized).
- **Agents.**
  - *Idea Agent:* writes hypotheses with observation, knowledge, justification and specification, using chain-of-thought and a seed hypothesis.
  - *Factor Agent:* writes several factors per hypothesis from an operator library over OHLCV leaves, filters them for complexity and alignment, and keeps a knowledge base of successes and failure modes.
  - *Eval Agent:* backtests and feeds results back to the Idea Agent.
  - Source: [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
- **Backtest setup.** — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
  - Qlib with a LightGBM of max depth 4 predicting next-day returns. Four base alphas: intraday return, daily return, 20-day relative volume and normalized range.
  - Top-k-dropout portfolio: hold the top 50 and drop 5.
  - Fees: CSI 500 is 5 bp to buy and 15 bp to sell; S&P 500 is 5 bp to sell.
  - Splits: train 2015–2019, validation 2020, test 2021-01 to 2025-01 (968 CSI 500 days and 1004 S&P days).
- **Headline results (2021–2024 test).** — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)

  | Market | AlphaAgent IC | ICIR | AR | IR | MDD | Best baseline AR |
  |---|---|---|---|---|---|---|
  | CSI 500 | 0.0212 | 0.1938 | 11.00% | 1.488 | −9.36% | LSTM 4.96% |
  | S&P 500 | 0.0056 | 0.0552 | 8.74% | 1.0545 | −9.10% | DeepSeek-R1 2.75% |

  The baselines include RD-Agent at CSI 500 IC 0.0113 and AR 0.78%, and AlphaForge at IC 0.0146 and AR 3.45%.
- **Ablation and decay claims.** — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
  - Hit ratio rose from 0.16 to 0.29 (+81%) with the regularizers on. A "hit" is an alpha in the top 5%, meaning annualized return >4.0% on CSI 500 or >1.8% on S&P 500.
  - Dev success rate rose from 0.75 to 0.83. Token efficiency is quoted as +23% in the ablation but "30% fewer tokens" in the abstract.
  - The three regularizers are **not ablated separately**.
  - In the decay figure (Fig. 4), on CSI 500 the IC of Alpha158, GP and RSI falls from about 0.022–0.036 to near zero over 2021–2024, while 15 AlphaAgent alphas hold IC ≈ 0.02 and RankIC ≈ 0.025.
- **Caveats.** — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
  - Base LLM: GPT-3.5-turbo by default. DeepSeek-R1 does best (S&P ICIR 0.0615, AR 9.19%).
  - RD-Agent was run with GPT-4-turbo. The reasoning-model baselines got best-of-10, while AlphaAgent and RD-Agent got 20 trials × 5 rounds, so the budgets are not matched.
  - OHLCV only, one predictor, one test window.

**R&D-Agent-Quant / RD-Agent(Q) (Li, Yang, Yang, Xu, Wang, Liu, Bian; Microsoft Research Asia + CMU/HKUST/Oxford)**
- **Dates and venue.** arXiv:2505.15155: v1 21 May 2025, v2 25 Sep 2025. The comments field says "NeurIPS 2025", which is author-supplied; I did not confirm it on a NeurIPS page. 42 pages. Code: github.com/microsoft/RD-Agent. — [arXiv abs](https://arxiv.org/abs/2505.15155)
- **Five units.** — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
  - *Specification:* S = (background B, data interface D, output format F, execution environment M = Qlib).
  - *Synthesis:* hypotheses drawn from action-conditioned history, grown into an "idea forest".
  - *Implementation:* Co-STEER. It builds a task-dependency DAG, retrieves from a knowledge base of task-code-feedback triples, and caps the inner loop at 10 iterations per task.
  - *Validation:* dedup against the SOTA library, then a Qlib backtest paired with the SOTA model.
  - *Analysis:* promotes improvements into the SOTA set, writes feedback and picks the next arm.
- **Dedup rule.** It computes the per-date IC between each new factor and each SOTA factor and averages over time. A factor is dropped if its maximum IC against any SOTA factor is **≥ 0.99**. Factors are fed to a fixed LightGBM, and the model arm uses a fixed "Alpha 20" factor set. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
- **Scheduler.** — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
  - A two-armed **contextual Thompson sampling** bandit with a linear reward r = wᵀx and a Gaussian posterior per arm (prior μ = 0, P = τ⁻²I).
  - Arms: {factor, model}.
  - 8-dimensional context: IC, ICIR, RankIC, RankICIR, ARR, IR, −MDD, SR.
  - The reward is "the actual improvement" observed, but the exact metric is not specified.
  - The appendix says exploration in one direction is capped.
- **CSI 300 results** (train 2008–2014, validation 2015–2016, test 2017-01 to 2020-08). — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)

  | System | IC | ICIR | ARR | IR | MDD | Calmar |
  |---|---|---|---|---|---|---|
  | RD-Agent(Q), o3-mini | 0.0532 | 0.4278 | 14.21% | 1.738 | −7.42% | 1.915 |
  | R&D-Factor, GPT-4o | 0.0489 | — | 14.61% | 1.684 | — | — |
  | Alpha158 | 0.0341 | — | 5.70% | 0.846 | — | — |
  | Alpha360 | 0.0420 | — | 4.38% | 0.673 | — | — |
  | AutoAlpha | 0.0334 | — | 4.00% | — | — | — |
  | TRA (best deep model on IC) | 0.0404 | — | 6.49% | — | — | — |

- **Factor count.** The abstract claims "over 70% fewer factors". The body says R&D-Factor used "only 22% of factors". — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
- **Newer out-of-sample test.** CSI 500 and NASDAQ 100, train 2008–2021, validation 2022–2023, test 2024 to Jun 2025. These splits were chosen so the test period postdates the LLM training cutoffs: GPT-4o 2023-10-01, o4-mini 2024-06-01. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)

  | System | CSI 500 IC | CSI 500 IR | NASDAQ 100 IC | NASDAQ 100 IR |
  |---|---|---|---|---|
  | RD-Agent(Q), o4-mini | 0.0288 | 2.17 | 0.0162 | 1.77 |
  | Alpha158 | 0.0192 | 0.25 | 0.0040 | 0.03 |

- **Controller ablation** (o3-mini, 12 h each). — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)

  | Controller | IC | ARR | Loops | Valid loops |
  |---|---|---|---|---|
  | Random | 0.0445 | 8.97% | 33 | 19 |
  | LLM-chosen | 0.0476 | 10.09% | 33 | 20 |
  | Bandit | **0.0532** | **14.21%** | 44 | 24 |

- **Cost.** Under $10 per run. Factor-only and model-only runs take 6 h each; the joint run takes 12 h. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
- **Reporting.** Five seeds with median ARR reported, and **no standard errors or confidence intervals**. The authors argue that consistency matters more than pointwise variance. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)

**AlphaGen (Yu, Xue, Ao, Pan, He, Tu, He; KDD '23 ADS; arXiv:2306.12964, 25 May 2023)**
- **Mechanism.** — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
  - PPO generates reverse-Polish-notation (RPN) token sequences, with invalid-action masking, a length cap and discount γ = 1, so longer formulas are not penalized.
  - Episode reward = performance (IC) of the **combination model after adding the new alpha**. Because the pool changes, the MDP is non-stationary.
  - The alpha pool persists across episodes.
- **Combiner.** — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
  - Linear weights fitted by gradient descent on MSE. The loss is computed from each alpha's mean IC and the pairwise mutual correlations, so the large z-scored values never need re-reading.
  - A new alpha enters with a random weight. If the pool exceeds its cap, the alpha with the smallest |w| is removed (Algorithm 1, line 8: p ← argminᵢ|wᵢ|).
  - Pool sizes tested: {10, 20, 50, 100}. Cost grows as O(k²) in mutual ICs, and returns diminish as the pool grows.
- **Data.** CSI 300 and CSI 500. Six features: OHLCV plus VWAP. Target: 20-day return. Train 2009–2018, validation 2019, test 2020–2021. Ten seeds. — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
- **Test results** (mean, with std over 10 seeds). — [arXiv PDF](https://arxiv.org/pdf/2306.12964)

  | Method | CSI 300 IC | CSI 300 RankIC | CSI 500 IC |
  |---|---|---|---|
  | AlphaGen | 0.0725 (0.0105) | 0.0806 | 0.0438 |
  | XGBoost | 0.0404 | — | — |
  | GP_filter (mutual IC < 0.7) | 0.0183 (0.0190) | — | — |
  | GP_top | 0.0078 | — | — |
  | PPO_top (single-alpha IC reward) | −0.0166 | — | — |

  A PPO agent rewarded on single-alpha IC "is prone to falling into local optima and thus overfitting on the training set, and basically stops searching".
- **Synergy evidence.** — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
  - In a 10-alpha set with combined IC 0.0511, alpha #1 has IC 0.0011 on its own, yet removing it and refitting drops the combination to 0.0447.
  - Alphas #2 and #6 have mutual IC 0.9746, yet 0.09317·f₂ − 0.07163·f₆ reaches test IC 0.0458, more than the sum of their individual ICs.
  - Conclusion: "neither the single alpha IC nor the mutual IC between alpha pairs is a good indicator of how well the combined alpha would perform".
  - On CSI 500, GP_filter falls below GP_top as the pool grows, so mutual-IC filtering can hurt.

**AlphaForge (Shi, Song, Zhang, Shi, Luo, Ao, Arian, Seco; AAAI 2025; arXiv:2406.18394, v1 26 Jun 2024, v5 12 Dec 2024)**
- **Mining stage.** — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
  - A differentiable *Predictor* P is trained by RMSE to predict the fitness of one-hot formula matrices. It is a **surrogate** for the expensive evaluation.
  - A *Generator* G (noise to Gumbel-softmax one-hot) is trained to maximize the frozen P, plus a **diversity loss** that penalizes one-hot and hidden-layer similarity between two sampled batches.
  - Fitness = |IC| if the formula is valid and its maximum |corr| with the existing zoo is below CORR′, else 0.
  - The zoo grows to a target size (100 in the case study).
- **Combination stage.** At each date t: — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
  - recompute each factor's IC and ICIR over the previous n days;
  - keep factors with IC > IC′ and ICIR > ICIR′;
  - sort by IC and take the **top N**;
  - fit **LinearRegression** on the latest data and predict.
- **Why linear and dynamic.** The authors say "nonlinear combination models are susceptible to overfitting in financial datasets". They argue fixed weights cannot adapt to factor "congestion" or style shifts, and they appeal to momentum in factor performance. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
- **Data.** Label Ref(VWAP,−21)/Ref(VWAP,−1)−1. Retrained annually over 2018–2022, five sessions in all; the first split is train 2010–2016, validation 2017, test 2018. Five seeds. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
- **Results (IC %).** — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)

  | Method | CSI 300 IC | CSI 300 RankIC | CSI 500 IC |
  |---|---|---|---|
  | AlphaForge (dynamic) | 4.40 (0.56) | 5.89 | 2.84 |
  | "Static" (same factors, AlphaGen-style fixed combiner) | 2.43 | — | 2.05 |
  | RL (AlphaGen) | 2.09 | — | 1.91 |
  | DSO | 2.55 | — | 1.38 |
  | GP | 1.29 | — | 0.37 |

- **Pool size.** Tested at {1, 10, 20, 50, 100}. The effect is **non-monotonic, best at 10**: "approximately 10 factors capture the most relevant price information". — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
- **Membership churn.** Of the 10 factors used on day 1, only 5 are used on day 2, and factor 3's weight flips from −0.00014 to +0.00168. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
- **Live trading.** The paper reports about 9 months of real-money trading against CSI 500. The thresholds n, IC′, ICIR′ and CORR′ are in the supplement, which I did not obtain. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)

**Alpha-GPT 1.0 / 2.0 and QuantAgent (Wang, Yuan, Guo et al., IDEA Research / HKUST-GZ)**
- **Alpha-GPT 1.0.** "Human-AI Interactive Alpha Mining for Quantitative Investment", arXiv:2308.00016: v1 31 Jul 2023, v2 20 Sep 2025. Now listed as **EMNLP 2025 System Demonstrations**. It is a prompt-engineering framework that turns a quant's ideas into alphas, positioned against genetic programming (GP). The abstract gives no numbers. — [arXiv](https://arxiv.org/abs/2308.00016)
- **Alpha-GPT 2.0.** "Human-in-the-Loop AI for Quantitative Investment", arXiv:2402.09746, 15 Feb 2024. Marked "Draft. Work in progress". It extends the loop to modeling and analysis, and the abstract gives no numbers. — [arXiv](https://arxiv.org/abs/2402.09746)
- **QuantAgent.** "Seeking Holy Grail in Trading by Self-Improving LLM", arXiv:2402.03755, 6 Feb 2024. — [arXiv](https://arxiv.org/abs/2402.03755)
  - Inner loop: refine answers against a knowledge base.
  - Outer loop: test them in the real market and write results back to the knowledge base.
  - The abstract gives no numbers.

**FAMA (Li, Song, Sun, Xu, Yu, Wen; Findings of ACL 2024, anthology 2024.findings-acl.233)**
- "Can Large Language Models Mine Interpretable Financial Factors More Effectively? A Neural-Symbolic Factor Mining Agent Model". — [ACL Anthology](https://aclanthology.org/2024.findings-acl.233/)
- Two parts: **Cross-Sample Selection (CSS)**, in-context learning from a diverse set of alphas to fight homogenization, and **Chain-of-Experience (CoE)**, reuse of past successful mining chains. The description of the two parts comes from the abstract and from citing papers. — [ACL Anthology](https://aclanthology.org/2024.findings-acl.233/), [Alpha Jungle citing FAMA](https://arxiv.org/html/2505.11122v3)
- Claims on S&P 500: RankIC +0.006 and RankICIR +0.105 over SOTA, a simulated 38.4% annualized return and a "Sharpe ratio of 667.2%". The last figure is implausible as written and probably means 6.67, which would itself be extreme; check before reuse. — [ACL Anthology](https://aclanthology.org/2024.findings-acl.233/)

**Navigating the Alpha Jungle (Shi, Duan, Li; Tsinghua IIIS; arXiv:2505.11122, May 2025; v3 12 Nov 2025; AAAI listing at ojs.aaai.org/…/37069)**
- **Search.** LLM proposals inside **MCTS**, using UCT, with any node expandable. The refinement dimension is sampled with weight toward the formula's weakest score. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37069)
- **Five-dimension evaluation.** Effectiveness (RankIC), stability (RankIR), turnover, diversity (maximum correlation with the repository) and an LLM-judged overfitting risk. Each non-LLM dimension is a **percentile rank against the effective-alpha repository**, and the overall score is their average. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
- **Frequent Subtree Avoidance (FSA).** It bans the top 3 frequent closed root subtrees from new formulas. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
- **Results.** CSI 300, 10-day target, set size 100, LightGBM: IC 0.0420, IR 0.94, versus AlphaAgent IC 0.0298, IR 0.33. In the ablation, CoT reaches IC 0.0434, ToT 0.0459 and MCTS+FSA 0.0549. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
- **Cost.** About $7.5 per run with Gemini-2.0-flash-lite (IR 1.27) versus about $74.4 with GPT-4.1 (IR 1.23). — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)

**Chain-of-Alpha (Lang Cao et al.; arXiv:2508.06312, 8 Aug 2025)**
- Two chains, a Factor Generation Chain and a Factor Optimization Chain, using backtest feedback. — [arXiv abs v2](https://arxiv.org/abs/2508.06312v2)
- **Status.** The arXiv v2 page (28 Aug 2025) says the paper was **removed by arXiv administrators** and lists Lang Cao as sole author. Search snippets list five authors (Cao, Xi, Liao, Yang, Cao). — [arXiv abs v2](https://arxiv.org/abs/2508.06312v2), [search result](https://arxiv.org/pdf/2508.06312)
- An alphaXiv summary reports CSI 500 ARR 0.1324 and IR 1.4178. **Unverified.** — [alphaXiv](https://www.alphaxiv.org/overview/2508.06312)

**AlphaSAGE (Chen, Ding, Shen, Huang, Guo, Liu, Zhang; arXiv:2509.25055, 29 Sep 2025, v3 19 May 2026; ICLR 2026 poster)**
- **Mechanism.** An RGCN structure-aware encoder and a GFlowNet generator, which samples roughly in proportion to reward rather than collapsing to one mode. — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3), [ICLR](https://iclr.cc/virtual/2026/poster/10006456)
- **Dense reward.** R_IC (|IC|) + R_SA (structure-behaviour alignment via k-nearest-neighbour distances in embedding space) + **R_NOV = 1 − max|IC| against a library of known alphas**. The SA and NOV weights are annealed, and an entropy regularizer is added to the loss. — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3)
- **Combiner.** It uses **AlphaForge's dynamic combiner**: recent-effective filter plus per-period linear regression. — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3)
- **Splits.** China: train 2010–2020, validation 2021, test 2022–2024. US: test 2018–2020. — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3)
- **Results.** CSI 300 IC 0.079, RankIC 0.094, AR 7.62%, SR 1.71, versus a plain GFlowNet at IC 0.048. S&P 500: SR 6.32 with MDD −4.2%, which looks implausibly high. — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3)

**Alpha-R1 (Jiang, Zhao, Sun, Sun, Li, Li, Jiang, Bai, Hua; SJTU / StepFun / FinStep; arXiv:2512.23515, 29 Dec 2025)**
- **Mechanism.** An 8B reasoning model (Qwen3-8B, trained with GRPO) **gates which factors are active** each date. It reads semantic factor descriptions plus a market state built from price and news. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Pool and combiner.** 82 factors filtered from Alpha101, with a fixed test set of 40 chosen by 2023–2024 RankIC. Selected factors enter a **fixed linear model** with coefficients estimated on 2020–2023. The top 10 stocks are held equal-weight, with 5-day holds and 0.1% cost per side. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Reward.** 5-day excess return × 100, scaled by a Claude-3.5-Haiku judge of reasoning quality, minus structural penalties. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Splits.** Train 2024-07 to 2024-12. Test 2025-01 to 2025-06, **only six months**. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Results.** CSI 300: AR 27.59%, SR 1.62. Zero-shot CSI 1000: AR 78.18%, SR 4.03. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Gating baselines.** Lasso gating reaches SR 0.20 and IC-momentum gating SR −0.80. Without RL, Alpha-R1 itself falls to SR −0.77. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Conflict with the NeurIPS listing.** A NeurIPS 2026 listing snippet reports 47.87% (S&P 500) and 40.57% (CSI 300) with SR 1.62 and 2.23. This conflicts with v1 and may reflect a revised version. — [NeurIPS listing](https://neurips.cc/virtual/2026/171189)

**Other 2025–2026 relatives (abstract-level only)**
- **AlphaAgentEvo** (ICLR 2026). Self-evolving agentic RL with a hierarchical reward. It claims a **4B model beats LLM-evolution methods running closed-source reasoning models**. — [ICLR](https://iclr.cc/virtual/2026/poster/10007685)
- **CogAlpha** (arXiv:2511.18850, Nov 2025, v4 Jul 2026). Code-level alphas with LLM mutation and recombination, tested on 5 datasets across 3 markets. — [arXiv](https://arxiv.org/abs/2511.18850)
- **AlphaMemo** (arXiv:2606.20625, 26 May 2026). — [arXiv](https://arxiv.org/abs/2606.20625)
  - It remembers edit *motifs*, taken from AST diffs, conditioned on the parent factor, rather than storing final factors.
  - It uses confidence-gated residual memory over a search-ledger prior, plus an "asymmetric veto" on high-confidence failure patterns.
  - It explicitly names the risk of overfitting from naively reusing past successes.
- **AlphaDiverse** (arXiv:2609.29014, 24 Sep 2026). — [arXiv](https://arxiv.org/abs/2609.29014)
  - Local Planner and Realizer agents, SFT-warm-started on diverse traces and then trained with joint GRPO for quality *and* diversity.
  - It names "research path collapse".
  - Research feedback uses only inner-period data, and a frozen final model is scored on a later outer period.
- **Hubble** (arXiv:2604.09601, Apr 2026). It concedes that it does **not** perform out-of-sample walk-forward validation. — [arXiv HTML](https://arxiv.org/html/2604.09601v1)
- **AlphaPROBE** (arXiv:2602.11917, Feb 2026). Retrieval plus on-graph biased evolution; details not checked. — [arXiv PDF](https://arxiv.org/pdf/2602.11917)
- **AlphaBench** (ICLR 2026). A benchmark of LLMs on factor generation, evaluation and search. It finds "persistent challenges in robustness, search efficiency, and practical usability". — [ICLR](https://iclr.cc/virtual/2026/poster/10008434)
- **Surveys.** "Towards Autonomous Formulaic Alpha Discovery: An Evolutionary Computation Perspective" (arXiv:2608.01789, 3 Aug 2026) proposes an 8-dimension evaluation that includes fitness reliability, residual alpha quality and robustness to non-stationarity. — [arXiv](https://arxiv.org/abs/2608.01789)
- An "Agentic Trading" survey (arXiv:2605.19337) counts autonomous alpha search as supported only when discovery is tied to out-of-sample or closed-loop evaluation (search snippet). — [arXiv HTML](https://arxiv.org/html/2605.19337v1)

### Inferences
- **Score marginal value against the current set.** AlphaGen's reward, the change in combination-model IC when the candidate is added, is the formal version of DOTA's "marginal value depends on the current set". The literature agrees the candidate should be scored by its marginal effect on the least-squares (LS) fitted ensemble, not by its standalone score. AlphaGen's pruning rule (drop the smallest |w| once the cap is exceeded) is a cheap, gate-free alternative to a paired t ≥ 2 swap test.
- **A bandit scheduler beats LLM choice, without the LLM.** RD-Agent(Q)'s controller ablation is the closest published analogue to "LLM hand-picking loses to the RF + Thompson surrogate". The bandit got +0.0056 IC and +4.1 pp ARR over LLM-chosen actions, and LLM-chosen actions were only slightly better than random. Their recipe is to let a statistical scheduler (Thompson sampling over coarse arms) pick *where* to search, and the LLM only *what* to write within the chosen arm.
- **AlphaForge is in effect a learned surrogate.** Its predictor network does the same job as DOTA's RF surrogate, and its dynamic combiner is DOTA's "≤12 members, LS weights" with periodic re-selection. Its finding that 10 of 100 factors is optimal and that membership churns daily suggests a fixed 12-member set frozen by a swap gate may be structurally suboptimal. Re-selecting per period or regime is the literature norm.
- **AlphaAgent's three regularizers map onto DOTA axes.**
  - Originality is correlation of a candidate's predictions with current members and prod. The `r2f0` residual-of-prod target already does this implicitly.
  - Complexity is the number of non-default recipe axes, feature-set size and selection depth.
  - Hypothesis alignment is making the LLM state *why* a recipe should add marginal value (for example horizon diversification) and checking that its realized correlation profile matches.
  - AlphaAgent never ablated the three separately, so there is no evidence which one carries the gain.

### Gaps
- No system reports confidence intervals across seeds for returns. RD-Agent(Q) explicitly declines to; AlphaGen and AlphaForge report the std of IC only.
- AlphaAgent's λ, α and β weights and its alpha-zoo contents are unreported. AlphaForge's IC′, ICIR′, CORR′ and lookback n are in a supplement I could not retrieve.
- The internals of Alpha-GPT 1.0 and 2.0 and QuantAgent (beyond their abstracts) were not verified, and none of the three gives numbers in its abstract.
- The NeurIPS 2025 acceptance of RD-Agent(Q) rests on the authors' arXiv comment only.
- No neutral, same-budget, same-data benchmark of these systems exists. Each paper's own method wins, and cross-paper numbers conflict: AlphaAgent shows RD-Agent at CSI 500 AR 0.78%, while Alpha Jungle shows AlphaAgent at IR 0.33.

---

## Q2. How systems build the "mega-alpha", and how they handle multicollinearity and pool size

### Takeaway
Almost every system combines factors **linearly**, either with fixed learned weights (AlphaGen), with a rolling OLS refit on the top N by recent IC (AlphaForge, AlphaSAGE), or as inputs to a GBDT (AlphaAgent, RD-Agent(Q), Alpha Jungle). Pools are small: an optimum around 10 in AlphaForge, and diminishing returns past a few dozen in AlphaGen. Redundancy is handled by hard correlation filters (0.7 in older GP work, 0.99 in RD-Agent(Q), CORR′ in AlphaForge) or by soft novelty rewards (AlphaSAGE, Alpha Jungle). AlphaGen shows that filters on mutual correlation can discard useful synergistic pairs.

### Cited Findings
- **AlphaGen.** — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
  - Linear combiner trained on MSE.
  - Cap K ∈ {10, 20, 50, 100}, with the minimum-|w| member evicted.
  - Mutual correlations are an *input* to the loss, not a filter.
  - Pairs with mutual IC 0.97 can be synergistic.
  - The "filter" baseline (pairwise mutual IC < 0.7) can underperform "top".
- **AlphaForge.** — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
  - Rolling per-date selection: thresholds on IC and ICIR over the previous n days, then top N by IC, then **LinearRegression** on recent data.
  - The best maximum N is 10 (non-monotonic in 1/10/20/50/100).
  - The authors prefer linear combiners because nonlinear ones overfit financial data.
  - Mining-time redundancy is controlled by a CORR′ cap against the zoo and a diversity loss.
- **AlphaSAGE.** It reuses AlphaForge's dynamic linear combiner and pushes novelty into the generator reward (1 − max|IC| against the library). — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3)
- **RD-Agent(Q).** New factors whose time-averaged IC against any SOTA factor is ≥ 0.99 are dropped, the survivors are concatenated into a fixed LightGBM, and the result is a set about 22–30% the size of the classical libraries. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
- **AlphaAgent.** Originality is enforced at generation time by AST-subtree similarity to the zoo, not by output correlation. The combination is LightGBM plus four base alphas. — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
- **Alpha Jungle.** Diversity enters the score as a percentile of maximum correlation with the repository. Combination is LightGBM or MLP over sets of 10, 50 or 100. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
- **Alpha-R1.** Combination is a fixed linear model fitted on an earlier block (2020–2023). The LLM chooses only the active subset, and Lasso and IC-momentum subset rules are its baselines. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Effective number of trials under correlation.** Bailey & López de Prado give an "implied number of independent trials" N̂ from M correlated trials, interpolating between N = 1 (ρ̄ → 1) and N = M (ρ̄ → 0) using the weighted average correlation ρ̄ (Appendix A.3). — [DSR paper PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- López de Prado & Lewis (2019, Quantitative Finance) propose clustering trials to estimate effective N [not re-fetched]. — [DOI](https://doi.org/10.1080/14697688.2019.1622311)

### Inferences
- **A cap of 12 fits the literature; freezing the set does not.** DOTA's cap is consistent with AlphaForge's optimum of 10 and AlphaGen's diminishing returns. The literature's members, though, are *re-selected* as performance drifts, and weights are refitted every period. A gate that only admits t ≥ 2 swaps makes the set path-dependent and frozen. AlphaGen's alternative is to always add the candidate, refit, and evict the smallest |w|, accepting the move only if the combined objective improves on a *proxy*.
- **Keep correlated candidates out of the gate, not out of the fit.** Because LS weights are fitted on a separate fit block, members that are highly correlated with incumbents can still be synergistic. AlphaGen's 0.97 pair is the example: spread-like differences of correlated signals. A hard redundancy filter can therefore discard value. A safer use of correlation is (a) the DSR/PBO effective-N computation, and (b) a soft novelty prior in the proposal step.
- **Least squares on a short fit block needs shrinkage.** With 12 members, a short fit block and strong collinearity, LS weights are high-variance. None of the agent papers uses ridge or shrinkage on the combiner, but AlphaForge's "linear only" argument points toward regularized LS (ridge, or non-negative weights). This is an inference, not a cited result.

### Gaps
- I found no agent paper that uses an explicit multicollinearity diagnostic (VIF or condition number) or shrinkage in the combiner. No paper studies weight instability of the mega-alpha across refits beyond AlphaForge's two-day anecdote.
- I found no paper on budgeted *which-to-train* selection with combination-aware value, the exact DOTA problem. AlphaGen and AlphaForge mine first and select later. RD-Agent(Q)'s bandit picks between two coarse arms, not among candidates.

---

## Q3. Overfitting controls in these systems, and the finance statistics that apply (DSR, PBO/CSCV, CPCV, Harvey–Liu–Zhu t > 3, White's Reality Check / Hansen SPA, purging and embargo)

### Takeaway
The agent papers rely on chronological train/validation/test splits, sometimes refitted annually or walk-forward, plus seeds, novelty and complexity penalties, and (in RD-Agent(Q)'s CSI 500 / NASDAQ 100 test and Alpha-R1) test windows that postdate the LLM's knowledge cutoff. **None** of the papers I read in full reports a deflated Sharpe ratio, a probability of backtest overfitting (PBO), combinatorial purged cross-validation (CPCV), a Reality Check or SPA test, or a multiple-testing-adjusted threshold for the many factors they try. The finance statistics literature supplies the missing tools. DSR deflates the selected Sharpe ratio by the expected maximum across N effective trials. PBO via combinatorially symmetric cross-validation (CSCV) measures how often the in-sample winner falls below the out-of-sample median. Harvey, Liu and Zhu argue for t > 3. Novy-Marx shows that combining the best k of n signals inflates results about as much as picking the best of nᵏ.

### Cited Findings

**Controls the agent papers do use**
- **RD-Agent(Q).** — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
  - Walk-forward model training; median of five seeds.
  - The LLM sees only the schema, not raw data or split dates.
  - "Health-check" modules verify factors are "leakage-free, non-trivial, and statistically meaningful".
  - The newer out-of-sample split is chosen to postdate the LLM training cutoffs.
  - No standard errors and no discussion of multiple testing.
- **AlphaForge.** Annual retraining with the prior year as validation. The authors cite overfitting to style shifts as the reason for the dynamic combiner. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
- **AlphaAgent.** Regularizers on originality, alignment and complexity, and a 4-year untouched test. — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
- **Alpha Jungle.** An LLM-judged "overfitting risk" dimension (qualitative, not validated) and FSA against homogenization. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
- **AlphaDiverse.** Inner-period feedback and an outer-period frozen evaluation, "avoiding test-set tuning". — [arXiv](https://arxiv.org/abs/2609.29014)

**Deflated Sharpe Ratio (Bailey & López de Prado, J. Portfolio Management 2014; SSRN 2460551)**
- **Expected maximum Sharpe.** Under N independent trials with Sharpe estimates of variance V: E[max SR] ≈ √V·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))), where γ ≈ 0.5772 is the Euler–Mascheroni constant. — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- **DSR.** It is the probabilistic Sharpe ratio (PSR) with the rejection threshold set to that expected maximum. PSR accounts for sample length, skewness and kurtosis. DSR therefore needs N, the variance of the trials' Sharpe ratios, T, skewness and kurtosis. — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- **Worked example.** A Treasury-auction strategy with Sharpe 2.5 over 5 years of daily data (T = 1250) has only about 90% probability of a true Sharpe > 0. It would have passed 95% after N = 46 trials, and after N = 88 if its returns had been Normal. The full input values are garbled in my text extraction. — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Appendix A.3 converts M correlated trials into an implied independent N̂ via the average correlation. — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- A related result: with 5 years of daily data, no more than about 45 independent configurations should be tried before a skill-less in-sample Sharpe of 1 is expected by chance (minimum backtest length) [not re-fetched]. — [Bailey, Borwein, López de Prado, Zhu, Notices AMS 2014](https://www.ams.org/notices/201405/rnoti-p458.pdf)

**Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado, Zhu)**
- **Version.** Revised Feb 2015; published in J. Computational Finance 2017 per standard citation [venue not re-fetched]. — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- **Algorithm.** — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
  1. Build a T × N matrix M of synchronous P&L, one column per configuration tried.
  2. Split its rows into an even number S of blocks.
  3. For every one of the C(S, S/2) combinations (S = 16 gives 12,870; the paper misprints it as "12,780"): take the best in-sample column, find its relative out-of-sample rank ω, and compute λ = ln(ω/(1−ω)).
  4. PBO = share of combinations with λ ≤ 0, that is, where the in-sample winner falls below the out-of-sample median.
- **Uses.** Reject models with PBO > 0.05, use PBO as a Bayesian prior, or weight strategies by (1−PBO). The same framework also gives performance degradation (in-sample vs out-of-sample regression), probability of out-of-sample loss and stochastic dominance. — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- **Limitations stated by the authors.** — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
  - Symmetric splits can obscure strongly autocorrelated performance when S is large.
  - All trials must be reported, because hidden trials bias PBO down (the file-drawer problem).
  - **For guided searches**, where an optimizer uses earlier iterations, "the columns of matrix M should be the final outcome of each guided search".

**Combinatorial Purged CV, purging and embargo (López de Prado, *Advances in Financial Machine Learning*, Wiley 2018)**
- **Purging and embargo.** Purging drops training samples whose label windows overlap the test set. An embargo also drops samples just after the test set. — [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) [not re-fetched]
- **CPCV.** The data is split into N groups and every combination of k test groups is tested, giving C(N,k) splits and multiple out-of-sample backtest paths, so the result is a distribution of out-of-sample Sharpe ratios rather than the single path a walk-forward test gives. — [fynance docs](https://fynance.readthedocs.io/en/latest/generated/fynance.data.combinatorial_purged_cv.html)
- **Splits vs paths.** One secondary source says N = 6, k = 2 gives "15 paths" ([claudeskills](https://claudeskills.info/skills/agiprolabs/claude-trading-skills/walk-forward-validation/)). By AFML's formula φ = (k/N)·C(N,k), that is 15 splits and 5 paths. Treat the secondary source as imprecise.
- **Limits.** Purging does not fix overfitting from testing many variants, which still needs DSR or a similar correction (secondary). — [LuxAlgo](https://www.luxalgo.com/library/concept/purged-cross-validation.md)

**Multiple-testing thresholds**
- **Harvey, Liu & Zhu.** "…and the Cross-Section of Expected Returns" (Review of Financial Studies 29(1):5–68, 2016; NBER w20592). A newly discovered factor needs **t > 3.0**, from a multiple-testing framework that allows correlated tests and missing data. The well-known count of 316 factors is not on the NBER landing page. — [NBER](https://www.nber.org/papers/w20592)
- **Novy-Marx.** "Backtesting Strategies Based on Multiple Signals" (NBER w21329, 2015). Combining the **best k of n** signals produces bias "almost as large as" (2016 version: "similar to") selecting the single best of **nᵏ** candidates. Random signals, each signed to predict positively in-sample, combine into highly "significant" backtests, so critical values "can be several times standard levels". — [NBER](https://www.nber.org/papers/w21329)
- **Haircut Sharpe ratios.** Harvey & Liu, "Backtesting" (J. Portfolio Management 42(1), 2015) [not re-fetched]. — [DOI](https://doi.org/10.3905/jpm.2015.42.1.013)
- **Data-snooping tests** [not re-fetched]:
  - White's Reality Check (Econometrica 68(5):1097–1126, 2000) bootstraps the best model's performance against a benchmark over the whole model universe. — [DOI](https://doi.org/10.1111/1468-0262.00152)
  - Hansen's SPA test (J. Business & Economic Statistics 23(4):365–380, 2005) studentizes and recenters to reduce the power loss from poor or irrelevant models. — [DOI](https://doi.org/10.1198/073500105000000063)
  - Romano & Wolf StepM (Econometrica 2005) identifies *which* models beat the benchmark. — [DOI](https://doi.org/10.1111/j.1468-0262.2005.00615.x)
  - The Model Confidence Set (Hansen, Lunde, Nason, Econometrica 2011) gives the set of models not significantly worse than the best. — [DOI](https://doi.org/10.3982/ECTA5771)
- **Sharpe-ratio standard errors under serial correlation.** Lo (2002), Financial Analysts Journal 58(4) [not re-fetched]. — [DOI](https://doi.org/10.2469/faj.v58.n4.2453)

### Inferences (for DOTA)
- **The 0.069 IC-select advantage is what DSR predicts.** Selecting on top-1% EV is an extreme-value problem. The expected maximum grows with √V·Φ⁻¹(1−1/N), and V (the cross-candidate variance of the noisy statistic) is far larger for a top-1% tail mean over ~30 val days than for IC. Selecting on a lower-variance statistic shrinks the selection bias.
- **DOTA's statistics map onto CSCV directly.** Columns are candidate ensembles or recipes; rows are val days, about 30 across 3 periods. PBO estimates how often the val-best ensemble falls below the median on held-out days.
  - With about 30 days, S = 6 or 10 blocks (whole days, stratified by regime) is more realistic than 16. This is my inference.
  - Per the PBO paper's guided-search rule, the *final* output of each agent episode or seed should be one column, not every intermediate candidate.
- **Use Novy-Marx at the ensemble level.** Picking ≤12 of hundreds of models is "best k of n". A t ≥ 2 gate on val is far below the critical values implied there. This explains the ensemble-level winner's curse, and suggests the gate should use DSR with effective N, or a Hansen-SPA-style test against prod, rather than raising t further.
- **Freezing and overfitting are two faces of one issue.** One fix, an inference, is to make the swap test *paired and cross-validated* on alternating val day blocks, with a deflated threshold that depends on how many swaps were tried. Paired by day is already done; DOTA's existing paired-by-day design is the right base.
- **Regime and day structure.**
  - Alternating-day val/test splits within the same regime are not independent samples. Adjacent days share the regime, and the storm period's eval volatility is 6.56 bp vs 1.62 bp in calm.
  - Purging and embargo are less about label overlap here (horizons are seconds), but each block should hold whole days. CSCV or CPCV partitions should be stratified by period, so that one regime does not dominate the in-sample winner.

### Gaps
- I could not verify a 2024 Knowledge-Based Systems study (Arian, Norouzi Mobarekeh, Seco) said to show CPCV beating walk-forward on PBO and DSR in synthetic data. The search did not surface it, so treat it as unverified.
- No agent paper applies DSR, PBO or SPA to its own mined set, so there is no empirical evidence on how much of the reported LLM-agent gains survive deflation.

---

## Q4. Evidence on LLM agents vs GP and RL baselines, and on out-of-sample decay of mined alphas

### Takeaway
Every LLM paper reports beating GP and RL, but the comparisons are usually budget-mismatched and self-run. Alpha Jungle gave LLM methods 1,000–3,000 evaluations against up to 600,000 for the others. AlphaAgent gave baselines best-of-10 against 20 × 5 rounds of its own. Different papers' rankings of the same rivals contradict each other. The independent evidence on decay comes from finance:
- published anomalies lose about 26% of their return out-of-sample and 58% post-publication (McLean & Pontiff);
- about 50% of in-sample Sharpe is lost out-of-sample (Falck, Rej & Thesmar), with **signal complexity** a significant predictor of decay;
- in-sample Sharpe barely predicts out-of-sample performance for 888 Quantopian algorithms (R² < 0.025), and more backtesting predicts a larger gap.

### Cited Findings
- **AlphaAgent decay figure.** GP-mined and Alpha158 factor ICs decay to near zero over 2021–2024 on CSI 500, while AlphaAgent's 15 alphas hold IC ≈ 0.02. This is the authors' own figure, with no independent replication. — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
- **AlphaGen.** Single-alpha-IC PPO overfits and stops exploring, with CSI 300 test IC −0.0166. GP keeps diversity but lacks synergy. — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
- **AlphaForge.** Under the same evaluation, GP reaches CSI 300 IC 1.29%, DSO 2.55%, RL 2.09% and AlphaForge 4.40%. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
- **Alpha Jungle.**
  - Budgets: LLM methods ran 1,000–3,000 searches, other methods up to 600,000 (200×).
  - The authors report that both in-sample and out-of-sample RankIC of the top 50 alphas improve over generations.
  - Non-LLM formulas were rated less interpretable.
  - Source: [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
- **RD-Agent(Q) vs AutoAlpha.** R&D-Factor (GPT-4o) reaches IC 0.0489 and ARR 14.61% on CSI 300, versus AutoAlpha (an evolutionary-search factor library) at IC 0.0334 and ARR 4.00%. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
- **Cross-paper conflicts.**
  - AlphaAgent's Table 2 has RD-Agent at CSI 500 AR 0.78%, IR 0.07. — [AlphaAgent](https://arxiv.org/html/2502.16789v2)
  - Alpha Jungle has AlphaAgent at IR 0.33 vs its own 0.94. — [Alpha Jungle](https://arxiv.org/html/2505.11122v3)
  - Each system loses in its rival's paper.
- **AlphaAgentEvo** claims a 4B RL-trained agent beats LLM-evolution methods using closed-source reasoning models (abstract). — [ICLR](https://iclr.cc/virtual/2026/poster/10007685)
- **Hybrid LLM–RL alpha discovery.** A Frontiers of Computer Science paper reports IC 75% above an RL baseline (search snippet, unverified). — [FCS](https://journal.hep.com.cn/fcs/EN/1159676948312023295)
- **AlphaBench** (ICLR 2026) reports "persistent challenges in robustness, search efficiency, and practical usability" for LLMs. — [ICLR](https://iclr.cc/virtual/2026/poster/10008434)
- **McLean & Pontiff** (J. Finance 2016). 97 predictors: returns are 26% lower out-of-sample (an upper bound on data mining) and 58% lower post-publication. — [search summary of JF paper](https://Www.Gwern.net/doc/economics/2016-mclean.pdf), [Ivey PDF](https://ivey.uwo.ca/media/3775549/pontiff.pdf)
- **Falck, Rej & Thesmar** ("Why and how systematic strategies decay", arXiv:2105.01380; Quantitative Finance 22(11), 2022). — [arXiv](https://arxiv.org/abs/2105.01380), [secondary summaries](https://www.evidenceinvestor.com/post/why-and-how-systematic-strategies-decay)
  - Out-of-sample performance is about 50% of in-sample, across 72 published US strategies (from secondary summaries).
  - Sharpe decay of newly published factors rises about 5 pp per year, and publication year explains 30% of the variance of decay.
  - **Number of operations in the signal** plus two outlier-sensitivity measures add about 15% explanatory power, a direct overfitting proxy.
- **Wiecki et al. (Quantopian).** 888 algorithms with at least 6 months live: backtest Sharpe gives R² < 0.025 for out-of-sample performance, more backtesting means larger in-sample/out-of-sample gaps, and a nonlinear classifier on backtest features reaches R² = 0.17. — [Quantpedia summary](https://quantpedia.com/quantopians-academic-paper-about-in-vs-out-of-sample-performance-of-trading-alg/)
- **AlphaMemo** measures decay as annual IC and RankIC on held-out years (per search snippet; no numbers verified). — [arXiv](https://arxiv.org/abs/2606.20625)

### Inferences
- **LLMs bring priors, not better selection.** The defensible claim is that LLM agents are *sample-efficient* thanks to priors: comparable or better results at about 200× fewer evaluations. They have not been shown to resist selection bias. This matches DOTA, where LLM hand-picking without priors loses to RF + Thompson. The LLM's comparative advantage is proposing from domain priors; selection and acceptance should stay statistical.
- **Penalize recipe complexity.** Falck et al.'s finding that complexity predicts decay independently supports AlphaAgent's complexity penalty. For DOTA, a complexity penalty on recipes is justified on finance-statistics grounds, not just by AlphaAgent's ablation.
- **Expect DOTA's sealed-test gains to shrink at OOS.** By the McLean–Pontiff and Falck et al. magnitudes, a deflation of roughly a quarter to a half is a reasonable prior when moving from val to the sealed OOS block. This is an inference from equity anomalies; tick-horizon signals may differ.

### Gaps
- No same-budget, independent benchmark of LLM vs GP vs RL alpha miners exists. AlphaBench is closest, but I could not read its numeric results.
- No paper measures alpha decay of LLM-mined factors beyond a single test window, or after deployment, except AlphaForge's anecdotal claim of 9 months of real money.

---

## Q5. Proxy objectives: IC/RankIC vs PnL/EV for selection vs evaluation

### Takeaway
The field mostly *selects and optimizes* on IC or RankIC, sometimes with ICIR, and *reports* portfolio returns. That is select-on-low-noise-proxy, score-on-money, matching DOTA's +0.069 finding. The papers' own tables show that IC rankings and return rankings disagree among the top candidates. AlphaGen reports that its framework, optimized only on IC, still delivers better backtests. The systems that reward returns directly (Alpha-R1) have six-month tests and fall apart without their RL component.

### Cited Findings
- **Who optimizes on what.**
  - AlphaGen is "only explicitly optimized against the IC metric" yet reports the highest IC and better top-k backtests. — [arXiv PDF](https://arxiv.org/pdf/2306.12964)
  - AlphaForge's fitness is |IC|, and its combiner selects by recent IC and ICIR. — [arXiv PDF](https://arxiv.org/pdf/2406.18394v5)
  - AlphaSAGE's main reward is |IC|. — [arXiv HTML v3](https://arxiv.org/html/2509.25055v3)
  - Alpha Jungle scores RankIC, RankIR, turnover and diversity as percentiles. — [arXiv HTML v3](https://arxiv.org/html/2505.11122v3)
  - RD-Agent(Q)'s bandit context mixes IC-type and return-type metrics. — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)
- **IC/return rank disagreements in RD-Agent(Q) Table 1 (CSI 300).** — [arXiv HTML v2](https://arxiv.org/html/2505.15155v2)

  | Pair | IC | ARR |
  |---|---|---|
  | Alpha360 vs Alpha158 | 0.0420 > 0.0341 | 4.38% < 5.70% |
  | MASTER vs LightGBM | 0.0215 < 0.0277 | 8.96% > 3.97% |
  | iTransformer vs TRA | 0.0270 < 0.0404 | 9.79% > 6.49% |
  | R&D-Factor GPT-4o vs o3-mini | 0.0489 < 0.0497 | 14.61% > 11.84% |

- **The same pattern in AlphaAgent Table 2 (CSI 500).** OpenAI-o1 has IC 0.0159 but AR 0.46%, while LSTM has IC 0.0175 and AR 4.96%. — [arXiv HTML v2](https://arxiv.org/html/2502.16789v2)
- **Training-loss mismatch.** LambdaRankIC (arXiv:2605.00501, 2026) notes the field trains on MSE but evaluates by RankIC or decile spreads. It proposes a listwise loss aimed directly at RankIC (search summary). — [arXiv PDF](https://arxiv.org/pdf/2605.00501)
- **IC sampling error.** A 1983 Financial Analysts Journal article ("the perils of the information coefficient") warns that ICs on fewer than several thousand observations carry significant sampling error (search summary). — [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/1983/problems-in-measuring-the-quality-of-investment-information-the-perils-of-the-information-coef)
- **Fundamental law.** Grinold (1989) gives IR ≈ IC·√breadth, so IC is the per-bet skill component of risk-adjusted return [not re-fetched]. — [DOI](https://doi.org/10.3905/jpm.1989.409211)
- **Direct return reward.** Alpha-R1 uses 5-day excess return as the GRPO reward, with reasoning-quality scaling. Without RL its gating has SR −0.77, and IC-momentum gating has SR −0.80. The test lasts only 6 months. — [arXiv HTML v1](https://arxiv.org/html/2512.23515v1)
- **Live-performance evidence (Quantopian).** Backtest Sharpe ratios are nearly uninformative about live Sharpe (R² < 0.025), while volatility, drawdown and other backtest features carry more signal. — [Quantpedia](https://quantpedia.com/quantopians-academic-paper-about-in-vs-out-of-sample-performance-of-trading-alg/)

### Inferences (for DOTA)
- **Make "select on IC, score on EV" the default.** DOTA's +0.069 result is consistent with standard practice and with extreme-value theory. IC uses every row; top-1% EV uses 1% of rows and is dominated by a few days. Selection should use the statistic with the smallest cross-candidate variance relative to its true spread. Only the *final* acceptance should be scored on EV, and it should be deflated (DSR-like) by the number of candidates considered.
- **Use a hybrid score at the top of the list.** IC and EV disagree among the leaders (the RD-Agent and AlphaAgent tables), so a hybrid helps. Options include IC as a prior with EV as a shrunk likelihood, for example a James–Stein / empirical-Bayes posterior of EV toward an IC-implied EV, or tail-aware ranking metrics (top-decile spread, or a LambdaRankIC-style listwise objective). These are design inferences, not tested results.
- **Rank within regime before averaging.** RD-Agent(Q) feeds both IC-family and return-family metrics into the bandit context, which also suits DOTA's surrogate features. With regime volatility differing 4×, the ranks or IC should be computed **within each period** and then averaged, rather than averaging raw EV, so the storm period does not dominate. DOTA's ratio-to-prod per period does part of this already.

### Gaps
- I found no paper that formally compares "select on IC, evaluate on PnL" with "select on PnL, evaluate on PnL" under controlled noise for alpha or ensemble selection. The evidence is indirect: rank disagreements in tables, extreme-value arguments and DOTA's own +0.069.
- No agent paper studies tail-conditioned objectives such as top-1% EV, as opposed to whole-cross-section IC, for selection.
