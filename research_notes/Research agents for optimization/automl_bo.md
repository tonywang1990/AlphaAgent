# Classical AutoML, Bayesian optimization / HPO, and LLM-as-optimizer hybrids (mechanism-level notes for AlphaAgent)

Scope note: notes were compiled on 2026-10-23. Each work is dated. A tag of **[verified]** means I read the claim this session in the primary source (abstract, HTML full text or PDF text). A tag of **[not re-fetched]** means the claim comes from the cited primary source as I know it, but I could not re-open the source this session, so treat the exact numbers with care. "DOTA case" refers to `/home/user/AlphaAgent/docs/case_studies/dota_ensemble_search.md` §4-8. The case setup: a budget of 96 trained recipes, an ensemble of ≤12 members, an RF + Thompson-sampling surrogate baseline, an LLM that steers the surrogate, greedy fill + swaps gated by paired t≥2, and no warm-started surrogate control yet.

---

## Q1. Mechanisms and lessons of the classical HPO/AutoML systems (Optuna/TPE + pruners, SMAC3, Hyperband/ASHA/BOHB, Vizier, Ax/BoTorch, auto-sklearn 1/2, AutoGluon, TPOT), and what they do about noisy objectives and overfitting the validation set

### Takeaway
The mature systems all separate three concerns. (a) A cheap, calibrated proposal mechanism: TPE density ratio, a random forest (RF) or Gaussian process (GP) surrogate plus an acquisition function, with random configurations interleaved. (b) A budget-allocation and evidence mechanism: successive halving or Hyperband, ASHA, and SMAC's intensification/racing. (c) A conservative final-selection or ensembling step: Caruana greedy ensemble selection with replacement, repeated k-fold bagging and stacking on out-of-fold predictions, and model-based incumbent selection. Noise and validation overfitting are handled mainly by **gathering more evidence for the incumbent (racing), using more robust resampling (repeated CV), and selecting with a regularized combiner**, not by a hard significance gate. The 2025 "overtuning" study shows validation overfitting is real but usually mild. It is severe in about 10% of runs and worst with small data and holdout validation.

### Cited Findings
**Optuna / TPE (2011; Optuna 2019)**
- TPE (Bergstra et al., NeurIPS 2011) splits observed trials by a quantile γ into "good" and "bad" sets. It fits a Parzen density to each, l(x) and g(x), and proposes the x that maximizes l(x)/g(x), which is equivalent to expected improvement under TPE's model. It handles conditional and categorical spaces natively, which is why it suits "recipe" spaces. — [Bergstra et al. 2011](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html); tutorial and ablations in [Watanabe 2023, arXiv 2304.11127](https://arxiv.org/abs/2304.11127) [not re-fetched]
- Optuna (Akiba et al., KDD 2019) is define-by-run: the search space is built dynamically in code. It pairs samplers (TPE by default; CMA-ES and GP among the alternatives) with **pruners** that stop unpromising trials from intermediate values: a median pruner, successive halving (ASHA) and Hyperband. — [Akiba et al. 2019, arXiv 1907.10902](https://arxiv.org/abs/1907.10902) [not re-fetched]
- OptunaHub has a "Meta-Learn TPE" sampler that weights source studies by task similarity as a guard against negative transfer. — [OptunaHub meta_learn_tpe](https://hub.optuna.org/samplers/meta_learn_tpe/) [verified via search snippet]
- In a 2026 budget-matched tabular study (8 PMLB tasks, 5 seeds), the gains from 12 to 40 evaluations were TPE +0.62 pp, random +0.76 pp and GP-BO +0.61 pp (all p≤1e-4). TPE, random and GP-BO were close to each other at 40 trials. — [Rodrigues & Vas, arXiv 2606.21641, 19 Jun 2026](https://arxiv.org/html/2606.21641v1) [verified]

**SMAC / SMAC3 (2011; JMLR 2022)**
- SMAC3 has four facades. SMAC4HPO uses a Sobol initial design, an **RF surrogate** and log-EI. SMAC4BB uses a GP with a Matérn-5/2 kernel and EI. SMAC4MF uses Hyperband intensification (BOHB-like) with an RF. SMAC4AC uses a default config as the initial design, an RF, log-EI and **aggressive racing**. — [Lindauer et al. 2022, SMAC3, JMLR 23; arXiv 2109.09831](https://arxiv.org/html/2109.09831v2) [verified]
- Under aggressive racing, "weaker candidates are evaluated on only a few instances, and more evidence on many instances is gathered only when a candidate could become the new incumbent." — [SMAC3 paper](https://arxiv.org/html/2109.09831v2) [verified]
- Original SMAC intensification works like this. Each iteration gives the incumbent an additional run on a new instance/seed. A challenger runs on a growing subset of the instance/seed pairs the incumbent has already seen, and is rejected as soon as its mean on that subset is worse. It replaces the incumbent only once it has at least as many runs and a mean at least as good. SMAC also **interleaves random configurations** to keep convergence guarantees and to protect against a misleading surrogate. — [Hutter, Hoos, Leyton-Brown, LION 2011](https://link.springer.com/chapter/10.1007/978-3-642-25566-3_40) [not re-fetched]
- RFs suit conditional and categorical spaces. In a 2025 reproducibility study, **SMAC's RF had the lowest NRMSE and highest R² as a pure regressor** among GP, SMAC and LLAMBO surrogates. The GP was best calibrated. — [Rychert et al., arXiv 2511.18891, Nov 2025](https://arxiv.org/html/2511.18891v1) [verified]
- On the multi-fidelity benchmarks in the SMAC3 paper, SMAC's multi-fidelity variant matched Hyperband early and led mid-run. Pure RF-BO caught up late. SMAC beat Dragonfly and, later in the run, BOHB. The paper reports these qualitatively from figures. — [SMAC3 paper](https://arxiv.org/html/2109.09831v2) [verified]
- Racing with statistical elimination (F-race, irace) keeps sampling candidates. It discards candidates only when a Friedman or paired t-test rejects them, so evidence keeps accumulating instead of freezing. — [López-Ibáñez et al. 2016, irace, Operations Research Perspectives](https://doi.org/10.1016/j.orp.2016.09.002) [not re-fetched]

**Hyperband / ASHA / BOHB (2016-2020)**
- Hyperband treats HPO as a pure-exploration, non-stochastic infinite-armed bandit. It runs successive halving at several aggressiveness levels ("brackets") over random configurations and reports "over an order-of-magnitude speedup" over BO competitors on deep-learning and kernel problems. — [Li et al., JMLR 18 (2018); arXiv 1603.06560](https://arxiv.org/abs/1603.06560) [verified]
- The Hyperband paper also popularized "random search at 2× budget" as a strong baseline that is competitive with several BO methods. — [Li et al. JMLR 2018](https://jmlr.org/papers/v18/16-558.html) [not re-fetched]
- ASHA is asynchronous successive halving. It promotes a configuration to the next rung whenever it is in the top 1/η of its rung, so workers never wait. It "scales linearly with the number of workers" and was shown on 500 workers. — [Li et al., MLSys 2020; arXiv 1810.05934](https://arxiv.org/abs/1810.05934) [verified]
- BOHB replaces Hyperband's random sampling with a TPE-like multivariate KDE fitted at the largest budget that has enough observations. It "consistently outperforms both Bayesian optimization and Hyperband on a wide range of problem types." — [Falkner et al., ICML 2018; arXiv 1807.01774](https://arxiv.org/abs/1807.01774) [verified]
- Multi-fidelity methods rely on **rank correlation between low and high fidelity**. Where it is weak, aggressive early stopping can kill configurations that are good late. In the 2025 overtuning study, early stopping (HEBO, 5-fold CV, up to 250 iterations) reduced overtuning only slightly (−0.275 on the log scale), "possibly because it stops good configurations too early." — [Schneider, Bischl, Feurer, AutoML Conf 2025; arXiv 2506.19540](https://arxiv.org/html/2506.19540v1) [verified]

**Google Vizier (2017; OSS 2022; GP-bandit 2024)**
- Vizier (KDD 2017) is Google's internal black-box optimization service. It features **transfer learning** and **automated early stopping**. — [Golovin et al. 2017, Google Research](https://research.google/pubs/pub46180/) [verified listing]
- Vizier's transfer learning builds a **stack of GP regressors, one per prior study, each trained on the residuals of the stack below it**. It assumes studies ran in a linear sequence, with earlier studies as priors. It works best when prior studies have many well-spaced points and the new study has little data. — [summary of the KDD paper, The Morning Paper](https://blog.acolyer.org/2017/10/02/google-vizier-a-service-for-black-box-optimization/) [secondary source; verified the summary]
- Vertex AI Vizier exposes a flag to turn transfer learning off. Otherwise, "vizier will automatically determine whether or not to use transfer learning." — [Vertex AI API docs](https://docs.cloud.google.com/dotnet/docs/reference/Google.Cloud.AIPlatform.V1Beta1/latest/Google.Cloud.AIPlatform.V1Beta1.StudySpec.Types.TransferLearningConfig) [verified]
- The current OSS Vizier default is a GP-bandit algorithm. The tech report (21 Aug 2024, rev. Dec 2024) says it has run "millions of optimizations" and claims robustness against industry baselines. Its abstract gives no numbers. — [Song et al., arXiv 2408.11527](https://arxiv.org/abs/2408.11527) [verified abstract]

**Ax / BoTorch (2019-2023)**
- BoTorch provides Monte-Carlo acquisition functions, including batch q-EI and q-Noisy-EI, on GP surrogates. — [Balandat et al., NeurIPS 2020; arXiv 1910.06403](https://arxiv.org/abs/1910.06403) [not re-fetched]
- **Noisy EI** integrates over the posterior of the function values at already-observed points, including the unknown true incumbent value. It does not plug in the noisy best observation, which avoids chasing a lucky incumbent. — [Letham et al., Bayesian Analysis 2019; arXiv 1706.07094](https://arxiv.org/abs/1706.07094) [not re-fetched]
- RGPE(NEI) applied the same NEI idea to transfer. — [Feurer et al. 2018/2022](https://ar5iv.arxiv.org/html/1802.02219) [verified]
- LogEI fixes vanishing gradients of EI-family acquisitions, which matters for optimizing acquisitions in large or flat spaces. — [Ament et al., NeurIPS 2023; arXiv 2310.20708](https://arxiv.org/abs/2310.20708) [not re-fetched]
- Overtuning authors suggest **selecting the final incumbent by posterior mean** rather than the raw best observed value in noisy BO, as a mitigation. It is suggested but not tested. — [Schneider et al. 2025](https://arxiv.org/html/2506.19540v1) [verified]

**auto-sklearn 1.0 (NeurIPS 2015) and 2.0 (JMLR 2022)**
- auto-sklearn 1.0 combines SMAC over a large conditional space (≈15 classifiers, 14 feature preprocessors, 4 data preprocessors, ≈110 hyperparameters) with two add-ons. The first is a **meta-learning warm start**: nearest-neighbour datasets in meta-feature space supply the first ≈25 configurations. The second is **post-hoc Caruana ensemble selection** over all models evaluated during search, so search effort is not wasted. — [Feurer et al., NeurIPS 2015](https://papers.nips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html) [not re-fetched]
- auto-sklearn 1 does post-hoc ensembling on a 33% holdout by default. Its authors reported that **stacking and gradient-free weight optimization overfit**, which made greedy ensemble selection (GES) the default in many AutoML systems. — [Purucker & Beel, AutoML Conf 2023](https://arxiv.org/html/2307.00286v1) [verified]
- auto-sklearn 2.0 drops meta-features in favour of a **greedily built static portfolio**. Each step adds the candidate pipeline that most reduces estimated error across 208 OpenML meta-datasets. The objective is monotone submodular, so greedy closes at least 1−1/e ≈ 63% of the gap to the optimal portfolio. It also adds **successive halving** and a choice of model-selection strategy: holdout, or 3-, 5- or 10-fold CV. The paper's reasons for dropping meta-features: they cost time, are undefined for categorical or missing data, and it is unclear which ones help. — [Feurer et al., JMLR 23 (2022); arXiv 2007.04074](https://ar5iv.labs.arxiv.org/html/2007.04074) [verified]
- Results: on 39 AutoML-benchmark datasets, relative error fell "by up to a factor of 4.5." auto-sklearn 2.0 at 10 min beat 1.0 at 60 min. — [JMLR abstract](https://jmlr.org/papers/v23/21-0992.html) [verified]

**AutoGluon-Tabular (2020; 1.x 2023-2026)**
- AutoGluon **does not centre on HPO**. It trains many model families with **repeated k-fold bagging** and **multi-layer stack ensembling**: later layers take out-of-fold predictions plus the original features, and a final layer does Caruana-style weighted ensembling. The paper argues this "uses allocated training time better" than searching for one best model. — [Erickson et al. 2020, arXiv 2003.06505](https://arxiv.org/abs/2003.06505) [verified abstract]
- Reported results: on 50 Kaggle/OpenML tasks it was faster, more robust and more accurate than TPOT, H2O, AutoWEKA, auto-sklearn and Google AutoML Tables. It often beat the best-in-hindsight combination of competitors. After 4 h of training on raw data it beat 99% of participants in two Kaggle competitions. These are self-reported. — [Erickson et al. 2020](https://arxiv.org/abs/2003.06505) [verified]
- AutoGluon 1.0 added `dynamic_stacking` to mitigate **"stacked overfitting"** and Zeroshot-HPO, a portfolio from TabRepo. Together they gave a **75% win-rate vs AutoGluon 0.8**. — [AutoGluon v1.0.0 release notes](https://auto.gluon.ai/1.1.0/whats_new/v1.0.0.html) [verified]
- How dynamic stacking works internally: as I understand it, it holds out data to check whether multi-layer stacking leaks or overfits, then decides whether to stack. — [GitHub issue #2779 / PR #3616 referenced in release notes](https://auto.gluon.ai/1.1.0/whats_new/v1.0.0.html) [unverified mechanism]
- AutoGluon's own docs place it first on TabArena and say AMLB 2025 rates AutoGluon 1.2 as state of the art. Both claims are vendor-reported. — [AutoGluon "What's new"](https://auto.gluon.ai/dev/whats_new/index.html) [verified as vendor claim]

**TPOT (2016)**
- TPOT uses genetic programming over tree-shaped sklearn pipelines. Selection is multi-objective (NSGA-II style) on CV score vs pipeline complexity, which works as a parsimony regularizer. — [Olson et al., GECCO 2016; arXiv 1601.07996](https://arxiv.org/abs/1601.07996) [not re-fetched]
- In AMLB (JMLR 2024), **TPOT had the worst average rank** among AutoML frameworks "in almost every setting." It was the only framework that did not beat all baselines in every setting. — [Gijsbers et al., AMLB, JMLR 25 (2024)](https://ar5iv.arxiv.org/html/2207.12560) [verified]

**Overfitting the validation set ("overtuning") and resampling**
- Overtuning (Schneider, Bischl, Feurer, AutoML Conf 2025, arXiv 24 Jun 2025) is HPO overfitting to validation error. A reanalysis of 7 benchmark collections (FCNet, LCBench, WDTB, TabZilla, TabRepo, reshuffling, PD1) found:
  - About 60% of runs had no overtuning.
  - About 10% were severe: no net test gain over the initial or default config.
  - In the TabRepo and reshuffling studies, more than 50% of runs overtuned and more than 15% severely.

  — [arXiv 2506.19540](https://arxiv.org/html/2506.19540v1) [verified]
- Determinants of overtuning (GLMM log-odds, relative to a holdout baseline):
  - Resampling: 5×5-fold CV −0.48, 5-fold CV −0.29, 5× holdout −0.26.
  - Dataset size: n=5000 −0.64 vs n=500.
  - Budget: more iterations meant more overtuning, with diminishing returns.
  - Optimizer: HEBO and SMAC3 slightly raised the odds of *any* overtuning vs random search but reduced its *magnitude*.
  - Recommendation: repeated CV is "the most effective and simplest" mitigation; adaptive resampling and racing are suggested.

  — [same](https://arxiv.org/html/2506.19540v1) [verified]
- Reshuffling the train/val split per configuration (Nagler, Schneider, Bischl, Feurer, May 2024, rev. Nov 2024) often improves test generalization. It "drastically" improves single-holdout HPO, making holdout competitive with CV at lower cost. — [arXiv 2405.15393](https://arxiv.org/html/2405.15393v2) [verified]
- Reshuffling raised overtuning slightly overall but strongly reduced it for holdout with ROC AUC. — [Schneider et al. 2025](https://arxiv.org/html/2506.19540v1) [verified]
- Classic warning: model-selection overfitting can be as large as the differences between learning algorithms, and selection bias contaminates performance estimates unless nested. — [Cawley & Talbot, JMLR 11 (2010)](https://jmlr.org/papers/v11/cawley10a.html) [not re-fetched]

### Inferences
- **DOTA's t≥2 gate vs SMAC's racing.** SMAC never requires significance to accept an incumbent. It requires *equal evidence* (the challenger is evaluated on the same instances or seeds as the incumbent) and a mean that is at least as good. It also keeps adding evidence to the incumbent, so a lucky incumbent is eventually exposed. A swap rule closer to SMAC/irace might help. It would treat val-day blocks (or bootstrap/CV folds of val days) as "instances" and race a challenger F′ against F on the same blocks. It would accept on a sequential test with a lower threshold once enough blocks agree, and re-verify the incumbent periodically. That replaces "a hard gate that freezes progress" with "more evidence before accepting." This is an inference; no source tests it on ensemble swaps.
- **Selection objective ≠ eval objective (DOTA §4.4)** echoes the overtuning finding that metric choice changes overtuning (ROC AUC and log-loss behave differently from accuracy). Choosing a smoother selection metric (IC) is an established lever.
- **Multi-fidelity is unexploited in DOTA real mode.** Fits cost minutes to hours, so ASHA or BOHB-style rungs could cut cost if a fidelity ranks recipes like the full fit: fewer training days, fewer trees, or a subsample. Low-to-high fidelity rank correlation must be checked first.
- **AutoGluon/TabRepo's lesson** is that ensembling diverse, fixed portfolio members usually beats tuning one model family. In DOTA terms, the "which 96 to train" question matters less than coverage and diversity of the bank. The marginal-gain signal should favour models that are *different* from F, not just strong.
- The case's observation that random overtook the surrogate at 64-96 models fits the general picture. Random search is a hard baseline at moderate budgets, and RF+Thompson sampling with a weak, set-dependent signal can over-exploit. SMAC deliberately interleaves random configurations for exactly this reason.

### Gaps
- I could not re-open the SMAC3 docs page on intensifier defaults (404) or Optuna's docs. The exact defaults (random-design probability, max config calls) are unverified.
- I could not verify the internal mechanism of AutoGluon's dynamic stacking (it holds out data to detect stacked overfitting) from a primary source this session.
- The NeurIPS 2020 BBO challenge paper ([Turner et al., arXiv 2104.10201](https://arxiv.org/abs/2104.10201)) is titled "BO is Superior to Random Search…". A search summary claimed ">100× sample-efficiency gains" for top submissions, but I could not confirm that figure in the paper text.

---

## Q2. Ensemble / subset selection: Caruana ensemble selection (with replacement, bagging, sorted init) and its overfitting; forward stepwise / Q²; combinatorial BO for subsets (BOCS, COMBO); submodular/greedy guarantees

### Takeaway
Caruana GES is a forward stepwise selection *with replacement* over a model library, scored on a hillclimb (validation) set. Its practical robustness comes from three regularizers: **with-replacement integer weights** (pseudo-discrete, sparse), **sorted initialization** from the N best models, and **bagged selection** (repeat over random subsets of the library and average). Its known failure is **overfitting a small hillclimb set**, which is fixed by larger or cross-validated hillclimb data. Gradient-free weight optimizers (CMA-ES) overfit more for some metrics (ROC AUC) unless their weights are normalized to look like GES's. Greedy guarantees (1−1/e, or 1−e^−γ with a submodularity ratio) hold for the *validation* objective only and say nothing about test.

### Cited Findings
- **Caruana et al. (ICML 2004)** build ensembles from libraries of thousands of models by forward stepwise selection on a hillclimb set. Metrics can be AUC, accuracy, RMSE, cross-entropy or F1. — [mlanthology abstract](https://mlanthology.org/icml/2004/caruana2004icml-ensemble) [verified abstract]; mechanism options in [Weka EnsembleSelection docs](https://weka.sourceforge.io/doc.packages/ensembleLibrary/weka/classifiers/meta/EnsembleSelection.html) [verified]
- Standard implementations expose the paper's three refinements: a **sort-initialization ratio** (start from the top-N library models), **"model bags"** (repeat selection on random subsets of the library and average), and **with/without replacement**. Some also add internal CV and pruning of the worst models before selection (pyensemble defaults: 3 CV folds). — [Weka docs](https://weka.sourceforge.io/doc.packages/ensembleLibrary/weka/classifiers/meta/EnsembleSelection.html); [pyensemble](https://github.com/dclambert/pyensemble) [verified]
- Overfitting behaviour in the 2004 paper, as I recall it:
  - Without replacement, hillclimb performance peaks and then *falls* as weaker models are forced in. With replacement, the curve flattens, because the best models can be re-added and weights become counts.
  - Bagged selection (about 20 bags, each over about half the library) reduces overfitting to the hillclimb set.
  - The ICDM 2006 follow-up ("Getting the most out of ensemble selection") showed that small hillclimb sets overfit. It used cross-validated or out-of-fold predictions so all training data serves as hillclimb data, and pruned the library to its best models.

  — [Caruana et al. 2004 PDF](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) and [ICDM 2006 combined long version](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf) [**not re-fetched**: server returned 503; exact numbers unverified]
- Bagged ensemble selection with out-of-bag hillclimbing (BES-OOB) is competitive with, and often better than, other ensemble strategies on binary classification. It beat stochastic gradient boosting on regression. — [Sun & Pfahringer, via Waikato repository](https://researchcommons.waikato.ac.nz/bitstreams/1f6becd3-ba58-4a8a-ac2b-fc6be654d0c9/download) [verified via search snippet]
- **GES as used in AutoML**: the arithmetic mean of an ensemble list built with replacement, so weights are counts/N. auto-sklearn stops after 50 iterations by default. The weights are "pseudo-discrete" and in practice sparse. There is no formal sparsity guarantee; the authors only *hypothesize* that this discreteness regularizes against overfitting. — [Purucker & Beel, "CMA-ES for Post Hoc Ensembling in AutoML", AutoML Conf 2023, PMLR 224](https://arxiv.org/html/2307.00286v1) [verified]
- **GES vs CMA-ES** (71 AMLB classification datasets, 10 folds, AutoGluon base models, about 13 base models on average):
  - For ROC AUC, CMA-ES overfits: its mean rank falls from validation to test (binary 1.02→1.83, multiclass 1.42→2.12) and GES wins on test.
  - For balanced accuracy, CMA-ES does *not* overfit (1.00→1.12, 1.03→1.25) and beats GES.
  - The fix is to **make CMA-ES weights GES-like**: trim weights ≤0.5/50 and round the rest to integer counts summing to 50 ("ExplicitGES"). That gives small val→test rank drift (1.63→1.70) and ≥ GES performance on ROC AUC.
  - Average non-zero ensemble sizes for ROC AUC: CMA-ES 12.9, ExplicitGES 6.3, GES 5.8.

  — [same](https://arxiv.org/html/2307.00286v1) [verified]
- Purucker et al. also proposed quality-(diversity-)optimization alternatives to GES (QO-ES, QDO-ES), population-based searches over ensembles. They argue GES's deterministic greedy path may be suboptimal. — [Purucker et al., AutoML Conf 2023, PMLR 224](https://proceedings.mlr.press/v224/purucker23b/purucker23b.pdf) [verified title and claim via search]
- **Number of GES steps** (TabRepo): the ablation found 15 Caruana steps best, and 25-200 steps nearly identical. The paper used 40. — [Salinas & Erickson, TabRepo, arXiv 2311.02971 (v3 Aug 2024)](https://arxiv.org/html/2311.02971v3) [verified]
- **Ensembling tuned single families** (TabRepo, random search over 200 configs, 4 h): ensembling lowered normalized error for every family, e.g. LightGBM 0.566→0.534 and MLP 0.634→0.594. "No model is able to beat state-of-the-art AutoML systems even with tuning and ensembling." — [TabRepo](https://arxiv.org/html/2311.02971v3) [verified]
- **Greedy guarantees**:
  - For maximizing a monotone submodular set function under a cardinality constraint k, greedy achieves at least (1−1/e) of the optimum. — [Nemhauser, Wolsey, Fisher 1978, Math. Programming 14](https://link.springer.com/article/10.1007/BF01588971) [not re-fetched]
  - For forward stepwise regression (choosing k predictors to maximize R²), greedy achieves (1−e^−γ), where γ is the **submodularity ratio**, which depends on the conditioning of the predictor covariance. Highly correlated members (low γ) weaken the guarantee. — [Das & Kempe, ICML 2011; arXiv 1102.3975](https://arxiv.org/abs/1102.3975) [not re-fetched]
  - auto-sklearn 2's portfolio construction uses exactly this monotone-submodular argument (≥63% of the gap closed; extending the greedy to 5·|P| steps closes about 99%). — [auto-sklearn 2.0](https://ar5iv.labs.arxiv.org/html/2007.04074) [verified]
- **Q² / forward stepwise caveat**: in QSAR, a high leave-one-out cross-validated q² obtained *after* variable selection does not imply external predictive power. This is the stepwise-selection version of the winner's curse ("Beware of q2!"). — [Golbraikh & Tropsha, J. Mol. Graph. Model. 2002](https://doi.org/10.1016/S1093-3263(01)00123-1) [not re-fetched; my interpretation that "Q2" in the brief means cross-validated Q² is an assumption]
- **BOCS** (Baptista & Poloczek, ICML 2018) is BO over binary vectors (e.g. include/exclude a member). It uses an "adaptive, scalable model that identifies useful combinatorial structure even when data is scarce" and pioneered "semidefinite programming" to optimize the acquisition. — [arXiv 1806.08838](https://arxiv.org/abs/1806.08838) [verified abstract]. Model details, from the paper as I know it: a Bayesian linear regression on first- and second-order interaction terms with a sparsity (horseshoe) prior, and Thompson sampling for acquisition [not re-fetched].
- **COMBO** (Oh, Tomczak, Gavves, Welling, NeurIPS 2019) represents the space as the graph Cartesian product of per-variable graphs and uses an ARD diffusion kernel to capture high-order interactions. A horseshoe prior on the ARD scales does variable selection, and the graph Fourier transform scales linearly. Evaluated on weighted MaxSAT and NAS. — [arXiv 1902.00448](https://arxiv.org/abs/1902.00448) [verified abstract]

### Inferences
- **The DOTA "greedy fill + gated swaps" maps onto GES *without* replacement plus a hard cap of 12**, which is the variant Caruana found most prone to overfitting. Cheap, literature-backed alternatives to a t≥2 gate:
  - (i) **GES with replacement** and a fixed step count (about 15-50 steps), so members enter as integer weights. A weak model then costs little, and the cap applies to *distinct* members.
  - (ii) **Bagged GES**: run GES on many bootstrap resamples of the val days (or random halves of the bank) and average the weight vectors. This smooths away lucky swaps without freezing progress.
  - (iii) **Cross-validated or out-of-block hillclimbing**: select on some val blocks and score on others, à la Caruana 2006 / BES-OOB.
  - (iv) **Sorted initialization**: start from the N best (or from prod). DOTA §8.3 already shows that fill-from-prod beats fill-from-empty, 1.128 vs 1.028.
- Purucker & Beel show that the overfitting of a weight optimizer depends on the metric and on weight discreteness. If DOTA ever optimizes continuous weights on `ev_drift`, snapping to GES-like integer counts is a tested regularizer.
- **Combinatorial BO is probably not needed for DOTA's ensemble step.** Scoring a candidate F is cheap (cached predictions), so the bottleneck is overfitting, not evaluation cost. BOCS and COMBO matter more for the *training* decision (which recipes to fit), where each evaluation is expensive and recipe axes interact. A BOCS-style pairwise-interaction model, or an RF, over one-hot axes is the right family there. DOTA §8.4 notes interactions matter ("a value weak on average can be the best cell paired with one other value"), which argues for a model with explicit pairwise terms or a tree model, not an additive one.
- **Submodularity caveat**: greedy guarantees bound the gap to the best *val* subset. With correlated members (low γ), even the val guarantee weakens, and none of these results address val→test transfer. The noise control has to come from evidence (more blocks, bagging), not from the optimizer.

### Gaps
- I could not fetch Caruana 2004 or 2006 (503 errors). The specific numbers (library ≈2000 models, bag fraction, sorted-init N, hillclimb-set sizes, overfitting curves) are unverified and should be re-checked before quoting.
- I found no study of BOCS or COMBO applied to *ensemble subset* selection specifically, as opposed to generic binary optimization.

---

## Q3. Transfer and warm-starting across tasks (meta-BO, RGPE / ranking-weighted GP ensembles, Vizier stacking, portfolios, HyperBO, PFNs4BO): evidence for gains and for negative transfer

### Takeaway
Transfer works best when it is **adaptive**: past tasks are weighted by how well they *rank* the new task's observations, and their influence decays as target data accumulates (RGPE, TAF, Vizier's residual stack). Static transfer, such as zero-shot portfolios and greedy portfolios, gives large gains on homogeneous benchmarks: TabRepo's portfolio plus ensembling beats AutoGluon, and auto-sklearn 2.0 cut error up to 4.5×. Learned priors (HyperBO, PFNs4BO) report ≥3× efficiency. Negative transfer is real but bounded in good designs. RGPE has a worst-case bound vs vanilla BO for simple regret, but individual variants still failed on some tasks, and naive warm-starts (WAC, ABLR, KL weighting) were "unreliable." The 2026 budget-matched study shows a key confound: **an apparent LLM "warm-start" advantage can be just a good default seed**. A warm-started classical control is required before crediting the LLM.

### Cited Findings
- **RGPE** (Feurer, Letham, Hutter, Bakshy; arXiv Feb 2018, rev. Oct 2022):
  - Mechanism: a weighted sum of per-task GP posteriors (so the ensemble is still a GP). Weights come from a **pairwise ranking loss** on the target's observations, using leave-one-out predictions for the target model. 1000 bootstrap samples are drawn and each sample's weight goes to its lowest-loss model.
  - "Weight dilution prevention" randomly drops base models with a probability that grows with the target data. **By the horizon only the target model remains.**
  - A worst-case bound (Theorem 1) makes vanilla BO a lower bound on simple-regret performance, up to a bounded slowdown factor.

  — [ar5iv 1802.02219](https://ar5iv.arxiv.org/html/1802.02219) [verified]
- RGPE results (normalized regret at 50 iterations, 6 benchmarks, 249 tasks): RGPE(TAF) was best or statistically tied on all six. Examples, RGPE(TAF) vs plain GP: SVM-4D 0.38 vs 0.68; NN-7D 1.55 vs 7.84; SVM-6D 0.45 vs 1.13. — [same](https://ar5iv.arxiv.org/html/1802.02219) [verified]
- Negative-transfer evidence inside RGPE:
  - The RGPE(NEI) variant had **final regret 2.20 vs 0.35 for plain GP on GLMNET-2D**, attributed to one dataset.
  - ABLR scored 2.95 on that benchmark.
  - Warm-start-by-initial-configs (WAC), ABLR and KL-weighting were "unreliable."
  - Using base-model *variance* hurt, because base models have zero variance at their own points and can repel search from good regions. Using only the target's variance was better.

  — [same](https://ar5iv.arxiv.org/html/1802.02219) [verified]
- **Vizier** stacks residual GPs across sequential studies. A related patent describes it as robust to uninformative priors. The service auto-decides whether to use transfer. — [The Morning Paper summary](https://blog.acolyer.org/2017/10/02/google-vizier-a-service-for-black-box-optimization/); [Vertex AI docs](https://docs.cloud.google.com/dotnet/docs/reference/Google.Cloud.AIPlatform.V1Beta1/latest/Google.Cloud.AIPlatform.V1Beta1.StudySpec.Types.TransferLearningConfig) [verified secondary]
- **Amazon SageMaker AMT**: simple warm-starting "typically works well when data is stationary, and can otherwise bias the surrogate model towards unpromising configurations". The paper lists robustness to "negative transfer" as future work. — [Perrone et al., arXiv 2012.08489](https://arxiv.org/pdf/2012.08489) [verified snippet]
- **HyperBO** (Wang et al., JMLR 25 (2024); arXiv 2021): pre-trains a GP prior (mean and kernel) on many related tasks with a KL-based loss, then **keeps it fixed** during BO, so the same data is not used for both training and the posterior. It reports bounded posterior predictions and near-zero regret without knowing the true prior, and finds good hyperparameters "at least 3 times more efficiently" than the best competing methods (authors' summary). — [JMLR v25 23-0269](https://jmlr.org/beta/papers/v25/23-0269.html); [arXiv 2109.08215](https://arxiv.org/html/2109.08215v6) [verified abstract]
- **PFNs4BO** (Müller, Feurer, Hollmann, Hutter; ICML 2023): a transformer trained on samples from a chosen prior acts as an in-context surrogate. It can mimic a GP or BNN, incorporate **user priors about where the optimum lies**, ignore irrelevant dimensions, and learn non-myopic acquisitions. Evaluated on HPO-B, Bayesmark and PD1. — [arXiv 2305.17535](https://arxiv.org/abs/2305.17535) [verified abstract; numbers not read]
- **πBO** multiplies the acquisition by a user prior over the optimum's location raised to a power that **decays over iterations**. A wrong prior is therefore washed out and convergence is preserved. PriorBand extends this to multi-fidelity. — [Hvarfner et al., ICLR 2022; arXiv 2204.11051](https://arxiv.org/abs/2204.11051); [Mallik et al., NeurIPS 2023; arXiv 2306.12370](https://arxiv.org/abs/2306.12370) [not re-fetched]
- **Zero-shot portfolios** (TabRepo: 1,310 configs × 200 datasets, leave-one-dataset-out):
  - A greedy portfolio plus ensembling reached normalized error **0.365 vs AutoGluon 0.389** at 4 h, statistically tied in a critical-difference diagram.
  - Others: portfolio without ensembling 0.434; auto-sklearn 2 0.455; FLAML 0.513.
  - A portfolio of size 3 beat every AutoML system except AutoGluon; size 15 beat AutoGluon.
  - Gains needed about 10 offline datasets to beat AutoGluon and saturated around 150.

  — [TabRepo, arXiv 2311.02971v3](https://arxiv.org/html/2311.02971v3) [verified]
- **auto-sklearn 2.0** chose a static portfolio over meta-feature kNN warm-starting, partly because of cost and robustness. Speed: 10 min of 2.0 beats 60 min of 1.0. — [ar5iv 2007.04074](https://ar5iv.labs.arxiv.org/html/2007.04074) [verified]
- **Default seed as confound** (Rodrigues & Vas, arXiv 19 Jun 2026; not peer-reviewed):
  - A fixed RF default scored 88.7% best-CV before any LLM call, vs random 83.7%, TPE 86.7% and GP-BO 86.3% at one evaluation. Every LLM advisor across a 7-model panel proposed the **same** default first.
  - The LLM's own proposals added **+0.40 pp on CV (CI [0.22, 0.62]) but −0.01 pp on test (p=0.92)**.
  - Seeding random search with the same default erased the LLM's lead: at 12 evals, seeded random was ahead by 0.37 pp (CI [−0.82, −0.02] for LLM − random).

  — [arXiv 2606.21641](https://arxiv.org/html/2606.21641v1) [verified]
- **LLM cross-task meta-information**: in the 2025 "in-context meta-learner" study (22 Kaggle tabular challenges, 10 models per dataset), meta-informed prompting (past tasks' models and scores in context) reached mean percentile rank 72.7, vs zero-shot 70.4, Context-Random 70.0 and HEBO-based random-hyperopt 65.7. Meta-information **sometimes hurt**: on mercedes 17.81 vs 36.44 zero-shot, and on influencers 76.84 vs 83.55. — [El Hili et al., arXiv 2510.26510 (v3 Feb 2026)](https://arxiv.org/html/2510.26510v3) [verified]

### Inferences
- **The missing DOTA control is exactly what the literature says to run.** At least two arms are needed to separate "memory" from "LLM reasoning":
  - (a) A surrogate warm-started on the same past episodes, either by pooling past (recipe, gain) data into the RF with an episode/task feature, or RGPE-style with one RF per past episode weighted by ranking loss on the current episode's points and decaying to the target.
  - (b) A **portfolio arm**: train first the top-k recipes from past episodes (TabRepo / auto-sklearn 2 style), then hand over to the surrogate.

  Rodrigues & Vas show that the "LLM warm-start" advantage can vanish once classical search gets the same seed. DOTA's large edge at 16 models (1.054 vs 0.994) is the regime where a seeded or portfolio control is most likely to close the gap.
- **The ranking loss is the right transfer signal for DOTA.** The surrogate is fitted on *marginal gain to the current F*, which is set-dependent and non-stationary across episodes. Absolute values transfer poorly, but rankings of recipe regions may transfer. RGPE's ranking-based weights and dilution prevention give a principled way to stop trusting memory when it misranks the new episode, which may explain non-monotone learning across episodes (DOTA §7).
- **Use priors that decay** (πBO-style) for LLM steering. If the LLM names a region, multiply the Thompson or acquisition score by a region prior whose weight decays with the number of in-episode observations. A wrong steer then costs a bounded amount.
- The El Hili results suggest **"Context-Random" ≈ zero-shot LLM**: random sampling from contextually relevant configurations nearly matched the LLM's picks. The value seems to lie in *where* to sample, not in point-wise picks, which matches DOTA §8.9.

### Gaps
- I did not find a large, dedicated empirical study that quantifies how often transfer-HPO methods hurt on *dissimilar* tasks (rates of negative transfer). The evidence is per-method ablations and outliers. A Rosenstein et al. (2005) style analysis for HPO was not found.
- I did not read the HyperBO or PFNs4BO results tables. Their "≥3×" and qualitative claims are the authors' summaries.
- I did not verify the exact definition of "Context-Random" in El Hili et al. (it appears to be random sampling from context-provided configurations).

---

## Q4. LLMs as optimizers or surrogates (OPRO, LLAMBO, LLM priors, LLM-proposed search spaces/regions) and 2025-2026 head-to-heads vs BO/random search: what division of labour works?

### Takeaway
Across 2023-2026 studies, LLMs help most where **priors** matter: cold start, tiny budgets (≤5-12 evaluations), semantic understanding of the task, reading logs, and choosing *regions* or *spaces*. They are **weaker than RF/GP surrogates as calibrated regressors** and tend to overfit validation when picking point by point. At equal budgets beyond about 10-20 evaluations, seeded classical search catches up or wins on clean HPO. The divisions of labour with the best evidence are:
- (1) the LLM proposes the seed, portfolio, search space or region, and a calibrated sampler (TPE, RF, GP, bandit) allocates evaluations inside it (HOLLM, LLINBO);
- (2) the LLM acts early and the statistical surrogate takes over as data accumulates (LLINBO, RGPE-like decay);
- (3) the LLM handles non-black-box information (logs, code, constraints) that classical HPO cannot see (AgentHPOBench).

### Cited Findings
- **OPRO** (Yang et al., Google DeepMind; arXiv 7 Sep 2023; ICLR 2024): the LLM optimizer receives a meta-prompt with previous solutions and their scores, sorted, and generates new candidates, which are scored and appended. Tested on linear regression and TSP, then mainly prompt optimization, where the best prompts beat human prompts by **up to 8% on GSM8K and up to 50% on Big-Bench Hard**. — [arXiv 2309.03409](https://arxiv.org/abs/2309.03409) [verified]
- **LLMs for HPO** (Zhang, Desai, Bae, Lorraine, Ba; arXiv Dec 2023, rev. Nov 2024): "within constrained search budgets, LLMs can match or outperform traditional HPO methods like Bayesian optimization." The paper also treats the model's *code* as a hyperparameter the LLM writes. — [arXiv 2312.04528](https://arxiv.org/abs/2312.04528) [verified abstract; budget and benchmarks not read]
- **LLAMBO** (Liu et al., ICLR 2024; arXiv 2402.03921): LLM-based zero-shot warm-start, an LLM surrogate (discriminative and generative) and an LLM candidate sampler, evaluated on Bayesmark and HPOBench. — [ICLR 2024 poster](https://iclr.cc/virtual/2024/poster/18743); [arXiv 2402.03921](https://arxiv.org/abs/2402.03921) [verified listing]
- **LLAMBO reproducibility** (Rychert, Spagnolo, Posashkov; arXiv 24 Nov 2025; Llama-3.1-70B instead of GPT-3.5; 30 tasks, 5 init + 25 trials, 5 runs):
  - Contextual warm-starts give lower early regret and lower variance than random, Sobol or LHS.
  - **As a regressor, SMAC's RF had the lowest NRMSE and highest R², and the GP was best calibrated. The LLM surrogate was weaker at low data and *overconfident*.**
  - The LLM candidate sampler had the lowest average regret vs TPE and random, with more balanced diversity (TPE "can collapse").
  - Gains are attributed to **"cross-task semantic priors"** from problem descriptions and hyperparameter names. Removing that text hurt most at few observations.
  - Smaller models (Gemma-27B, Llama-3.1-8B) produced malformed outputs and surrogate scores uncorrelated with performance.

  — [arXiv 2511.18891](https://arxiv.org/html/2511.18891v1) [verified]
- **Budget-matched tabular study** (Rodrigues & Vas; arXiv 19 Jun 2026; 8 PMLB tasks × 5 seeds; LLM advisor = Claude Haiku with 12 iterations vs random/TPE/GP-BO with 40):
  - The LLM's apparent lead came from its first proposal, a strong RF default. Its own later proposals added +0.40 pp CV and **−0.01 pp test**.
  - With the same default seed, random search tied or beat it: −0.09 pp (n.s.) at 5 evaluations, and ahead by 0.37 pp at 12.
  - Unseeded TPE tied at 12 (p=0.10) and passed it by 40.
  - On one task (vehicle) the LLM stayed at the default, 73.3% vs random 82.4%.
  - Its rule-based confidence filter saved about 33% of wasted evaluation time without accuracy gain and "needs no LLM at all."
  - Recommendation: "seed classical search with a sensible default."

  — [arXiv 2606.21641](https://arxiv.org/html/2606.21641v1) [verified]
- **AgentHPOBench** (Huai et al., Fudan; arXiv 31 Jul 2026):
  - Setup: 30 executable ML research repos in 7 categories. After a reference baseline run, agents make **5 sequential interventions**, each at about 10% of the original training budget. The score is the **final** (not best) configuration.
  - Limited-budget results, mean bounded normalized score (MBNS): random search −0.034, TPE −0.113, BOHB variant 0.018. Claude Sonnet 4.6 0.407 (beat the baseline on 23/30 tasks), GPT-5.5 0.305, DeepSeek-V4-Pro 0.175, Qwen3-32B 0.148.
  - Small models (Gemma2-2B −0.001, Llama-3.1-8B 0.030) were roughly at the level of the classical baselines.
  - Classical HPO saw only configs and the target metric. **Agents also saw task context, auxiliary metrics and logs.**
  - Removing intermediate feedback cut Qwen3-32B's MBNS from 0.148 to 0.052.
  - Agents showed "nonmonotonic refinement" and often lost earlier gains.
  - API agents were run once, while open models and baselines got 3 seeds.

  — [arXiv 2607.29626](https://arxiv.org/abs/2607.29626) (PDF text read) [verified]
- **In-context meta-learning** (El Hili et al.; arXiv Oct 2025, rev. Feb 2026): with 10 models per dataset on 22 Kaggle tasks, zero-shot LLM recommendation (70.4 p-rank) and meta-informed (72.7) beat HEBO-based search (Random-Hyperopt 65.7, LGBM-Hyperopt 61.8). Context-Random reached 70.0. — [arXiv 2510.26510](https://arxiv.org/html/2510.26510v3) [verified]
- **HOLLM** (Schwanke, Ivanov, Zela et al.; arXiv May 2025; ICLR 2026) is the closest published analogue to "LLM names a region, a calibrated mechanism allocates":
  - It builds a **KD-tree partition** of the space over evaluated points.
  - Each leaf is a "meta-arm" with a **bandit utility** that balances observed value against region size and uncertainty, and leaves are chosen stochastically.
  - The **LLM proposes candidates only within the chosen subregion**, which "significantly reduces LLM sampling difficulty."
  - It matches or beats leading BO and trust-region methods and "substantially" beats global LLM sampling.

  — [arXiv 2505.21372](https://arxiv.org/html/2505.21372v2); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html) [verified abstract; numbers not read]
- **LLINBO** (Chang, Azvar, Okwudire, Al Kontar; arXiv May 2025) pairs an LLM with a GP, motivated by the LLM's lack of "explicit surrogate modeling and calibrated uncertainty."
  - The LLM is used for **early exploration**, and the GP takes over for exploitation.
  - There are three mechanisms with theoretical guarantees: Transient (hand-off over time), Justify (the LLM's proposal must be justified against the GP's acquisition) and Constrained.
  - Evidence: a 3D-printing proof of concept.

  — [arXiv 2505.14756](https://arxiv.org/html/2505.14756v2) [verified abstract; performance claims not checked]
- Other 2025-2026 hybrids found but not read: an LLM-TPE hybrid sampler (SLLMBO, [arXiv 2410.20302](https://arxiv.org/html/2410.20302)); language-guided BO for LoRA HPO ([arXiv 2602.11171](https://arxiv.org/pdf/2602.11171)); preference-guided LLM-in-the-loop BO ([arXiv 2605.17976](https://arxiv.org/pdf/2605.17976)). [not read]
- Learned-optimizer precursor: OptFormer learns an HPO policy and prior from Vizier tuning traces. — [Chen et al., NeurIPS 2022; arXiv 2205.13320](https://arxiv.org/abs/2205.13320) [not re-fetched]

### Inferences
- **The DOTA results line up with this literature.**
  - Naive LLM hand-picking lost to RF+Thompson sampling. This matches Rychert (the LLM surrogate is a worse regressor and overconfident) and Rodrigues & Vas (LLM proposals raised CV but not test).
  - The LLM that *steers the surrogate by naming a region* won. This is the HOLLM/LLINBO pattern: an LLM prior on *where*, with a calibrated mechanism deciding *which* and *how many*.
  - The win came with cross-task memory. This matches LLAMBO's gains from "cross-task semantic priors" and the meta-informed > zero-shot result, but also the warning that meta-information can hurt on some tasks.
- **A defensible division of labour for AlphaAgent**:
  1. **Memory/LLM → initial portfolio and region priors**: name the regions, give each a prior weight and expected gain, and record its rationale.
  2. **Calibrated sampler → allocation inside regions**: Thompson sampling or EI on an RF over one-hot axes with pairwise interactions, plus a fixed fraction of random draws.
  3. **A bandit over regions (HOLLM-style)**, so that LLM-named regions compete with surrogate-proposed and random regions on observed marginal gain. This makes a bad steer self-correcting.
  4. **A decaying LLM prior weight** (πBO/LLINBO-Transient), so the surrogate dominates as in-episode data accumulates.
  5. **The LLM's comparative advantage in non-black-box signals**: reading diagnostics such as per-regime breakdowns and correlation with F, spotting bugs (DOTA §8.8), and proposing *new axes or recipes* (search-space edits), which no classical sampler can do.
- **Benchmark-design lesson**: AgentHPOBench's strong agent result comes with a very small budget (5 steps), richer inputs for agents than for baselines, single runs for API agents, and final-step scoring. Rodrigues & Vas show that equalizing the seed removes much of the edge. AlphaAgent evaluations should (i) give baselines the same seeds and memory, (ii) report curves over budget, (iii) use paired multi-seed tests, and (iv) score held-out test, not validation. DOTA's protocol already does most of this.

### Gaps
- I found no peer-reviewed 2025-2026 head-to-head of *LLM agents vs AutoGluon/auto-sklearn* on AMLB or TabArena at equal compute. Existing agent papers (AutoML-Agent, TML-Bench, LightAutoDS-Tab) use their own setups. — [TML-Bench, arXiv 2603.05764](https://arxiv.org/pdf/2603.05764); [AutoML-Agent, arXiv 2410.02958](https://arxiv.org/pdf/2410.02958) [titles verified only]
- I did not find a study of **LLM-proposed search spaces** (as opposed to points) measured against expert spaces at equal budget. Zhang et al. 2023's "code as hyperparameter" is the closest. This remains a gap.
- I did not read HOLLM's or LLINBO's numerical results.

---

## Q5. Benchmarks (HPOBench, YAHPO Gym, HPO-B, AMLB, TabRepo/TabArena) and how big the differences between methods are vs random search

### Takeaway
The differences between good HPO and AutoML methods are **usually small relative to noise**, and random search (especially at 2× budget) is a hard baseline. Large effects come from **what is searched** (portfolios, ensembling, stacking, priors and transfer) more than from the choice among BO, TPE and SMAC. In AMLB the top frameworks are "very competitive with each other," and only AutoGluon and LightAutoML were significantly better than the baselines across all settings. In the 2026 tabular study, TPE, random and GP-BO gained similar amounts from 12 to 40 evaluations. Statistical protocols (multiple seeds, paired tests, critical-difference diagrams) are standard because the method effects are near noise.

### Cited Findings
- **AMLB** (Gijsbers et al., JMLR 25(101), 2024): 9 frameworks, 71 classification and 33 regression tasks, 1 h and 4 h budgets on 8 vCPUs.
  - AutoGluon (best quality) had the best average rank almost everywhere but was **not significantly better than the second-best**.
  - Only AutoGluon(B) and LightAutoML were significantly better than the baselines across all settings. Every framework except TPOT ranked above all baselines; the strongest baseline was a tuned random forest.
  - Results were "very similar" across 1 h and 4 h.
  - Failures came mainly from memory and time limits. Inference speed differed by orders of magnitude.

  — [ar5iv 2207.12560](https://ar5iv.arxiv.org/html/2207.12560); [JMLR](https://jmlr.org/papers/v25/22-0493.html) [verified]
- **TabRepo** (2023/2024): normalized errors at 4 h: portfolio+ensemble 0.365, AutoGluon 0.389, auto-sklearn 2 0.455, LightAutoML 0.466, FLAML 0.513, H2O 0.526. Tuned+ensembled single families (e.g. LightGBM 0.534) do not reach AutoML systems. — [arXiv 2311.02971v3](https://arxiv.org/html/2311.02971v3) [verified]
- **Rodrigues & Vas 2026**: at 40 trials, TPE, random and GP-BO were within about 0.15 pp of each other in gains from 12→40 evaluations. A single strong default (88.7%) beat all methods' first evaluations. — [arXiv 2606.21641](https://arxiv.org/html/2606.21641v1) [verified]
- **AgentHPOBench 2026**: at 5 interventions, random search (−0.034), TPE (−0.113) and BOHB (0.018) were all near zero improvement over the baseline, and random search had the highest baseline win-rate among them (48.9%). Tiny budgets make classical methods look like noise. — [arXiv 2607.29626](https://arxiv.org/abs/2607.29626) [verified]
- **NeurIPS 2020 BBO challenge** (Turner et al., arXiv Apr 2021): the paper's title claims "Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning." Objectives were validation performance of standard models on real datasets, with hidden functions and no human intervention. — [arXiv 2104.10201](https://arxiv.org/abs/2104.10201) [verified abstract; effect size not verified]
- **Overtuning** (2025): with 500 random-search iterations, about 38.5% of runs improved test by <0.001 over the initial config, and about 10% were severely overtuned. Much of the "improvement" from HPO on small tabular data is within noise. — [arXiv 2506.19540](https://arxiv.org/html/2506.19540v1) [verified]
- **Benchmark suites (for AlphaAgent offline testing)**:
  - HPOBench (Eggensperger et al., NeurIPS 2021 D&B): containerized raw, tabular and surrogate benchmarks, with multi-fidelity. — [arXiv 2109.06716](https://arxiv.org/abs/2109.06716) [not re-fetched]
  - YAHPO Gym (Pfisterer et al., AutoML Conf 2022): surrogate-based, multi-fidelity and multi-objective scenarios; it argues tabular benchmarks bias conclusions. — [arXiv 2109.03670](https://arxiv.org/abs/2109.03670) [not re-fetched]
  - HPO-B (Pineda Arango et al., NeurIPS 2021 D&B): built from OpenML runs, designed for **transfer-HPO** evaluation. — [arXiv 2106.06257](https://arxiv.org/abs/2106.06257) [not re-fetched]

  Exact problem counts were not verified this session.
- **Random-search baselines**: the Hyperband paper's "random search at 2× budget" comparisons are commonly cited as showing several BO methods only modestly beat random search. — [Li et al., JMLR 2018](https://jmlr.org/papers/v18/16-558.html) [not re-fetched]

### Inferences
- The DOTA numbers have the same character. Between arms at a given budget the differences are about 0.02-0.06 on a test SE of about 0.02 per round, and random overtakes the surrogate at 64-96 models. Expect method differences to be small. Seeds, paired tests and equal-information baselines matter more than optimizer choice.
- The biggest DOTA lever, fill-from-prod (1.128 ceiling vs 1.028 from empty), resembles the benchmark finding that **initialization, portfolio and ensembling choices dominate optimizer choice**. AlphaAgent should spend design effort on what to start from, what to ensemble and how to regularize selection, and treat the sampler as a commodity: an RF+TS / TPE / random mix.
- For a general AlphaAgent benchmark harness: use YAHPO or HPOBench surrogates for cheap optimizer ablations, HPO-B or TabRepo for transfer and portfolio ablations, and AMLB or TabArena-style held-out test protocols for end-to-end claims. Report curves vs budget with equal seeds and memory given to baselines.

### Gaps
- I did not obtain AMLB's numeric gap between frameworks and the tuned-RF baseline (the paper reports ranks and scaled boxplots), nor the AMLB 2025 update results (only AutoGluon's vendor claim).
- I did not verify the BBO-challenge effect size vs random search, or the exact sizes of HPOBench, YAHPO and HPO-B.
- I found no benchmark of *ensemble-member selection under a training budget with a member cap*, i.e. DOTA's joint "what to train + what to keep" problem. The closest analogues are TabRepo (portfolio + GES over precomputed models) and auto-sklearn (SMAC + post-hoc GES).
