# Response to the faculty review — Premier League Match Predictor

**Decision received:** accepted with major revisions. This file maps each required modification to what was done and where the
evidence is. All numbers come from one consistent run of the pipeline (`requirements-lock.txt`).

## Priority recommendation — what the evidence says
Previous-season FPL squad features do **not** consistently improve held-out match probabilities over strong football baselines.
Over five walk-forward seasons (2021-22 → 2025-26) the mean log loss is 0.979 (Elo + form, LR), 0.981 (Elo only, LR),
0.982 (Elo + form + FPL, RF), 0.984 (Elo + form + FPL, LR) and 0.982 (Dixon-Coles). Every FPL-related paired difference has a
95% bootstrap interval that includes zero, and adding FPL to Elo + form was better in only 1 of 5 seasons for Logistic
Regression and 2 of 5 for Random Forest. Only the bookmakers are significantly better (about −0.024 log loss).
The work is therefore presented as a **reproducible comparative study**, not a new prediction algorithm.

## 1. Clarify novelty
- **What was tried:** a team-level squad-strength representation from the official FPL player data of the *previous* season
  (average price, points and ICT index of the 11 most-used players, points by position, total squad value, goals, assists),
  with promoted teams flagged and filled with the average of the three weakest teams. It is built only from past seasons.
- **How it differs from existing approaches:** Elo ratings (Hvattum & Arntzen, 2010) and Dixon-Coles goal models (1997) use
  only match results; FPL prices and points act as a cheap, public stand-in for player quality (FIFA/EA FC ratings were not
  reachable). The contribution is the *test* of whether such a stand-in adds information beyond Elo and rolling form.
- **Finding:** it does not, with these data (see above and `results/paired_differences.csv`).
- No claim of a new algorithm is made anywhere in the README or the app.

## 2. Feature ablations (identical splits and metrics)
`src/06_ablation.py` → `results/oof_predictions.csv`; metrics in `results/ablation_summary.csv`, `results/ablation_per_season.csv`.
Sets: A Elo only, B Elo + rolling form, C Elo + form + FPL (the 27 compact features), D Elo + FPL (extra). Logistic Regression and
Random Forest, expanding window, every model sees exactly the same train/test rows per season.

## 3. Football-specific baselines
`src/dixon_coles.py`: time-decayed Dixon-Coles Poisson model with low-score correction, refitted monthly on past matches only.
Also reported: always-home-win, Elo-only logistic regression, and bookmaker implied probabilities (reference only, never a feature).

## 4. Data audit and leakage
`src/audit_data.py` (all checks pass; output in `results/data_audit.txt`):
- Counts reconciled: raw 3,820 = 380 (2016-17 history) + 3,440 modelled; modelled = 2,660 development (2017-18 → 2023-24)
  + 760 test (2024-25, 2025-26) + 20 live 2026-27 games (never trained or tested on). The deployed model is fitted on the
  3,420 completed matches of 2017-18 → 2025-26. The proposal's two descriptions (2017-18 → 2025-26 for modelling, 2016-17 →
  2026-27 for the raw file) are both correct and refer to these two different things.
- Only completed matches are used (decided by the presence of a result; unplayed fixtures are dropped).
- Elo is a pre-match rating; the Elo in the model table equals the cleaned pre-match Elo for all 3,440 matches.
- No same-match statistic, result or bookmaker column is a feature; stored features equal features recomputed from strictly
  earlier matches (all 27 model features, 400 sampled matches in the audit and all 3,440 in a one-off check; the training and the
  live prediction code are one shared module, `src/features.py`); FPL features come from the previous season; fill values use development seasons only.

- Step 1 is reproducible: `src/01_check_raw_data.py` verifies the raw files against SHA-256 checksums (`data/raw/MANIFEST.csv`).

## 5. Probability and draw evaluation
`src/07_probability_eval.py` → `results/probability_report.txt`, `results/calibration_by_class.csv`, figures 7 and 8.
Reported: multiclass log loss, Brier score, expected calibration error, calibration curves per class, draw precision and recall
(argmax and a draw rule with a threshold chosen on 2021-22 → 2023-24 only), accuracy with the draw rule, per-season mean and
standard deviation of log loss, and paired bootstrap confidence intervals. Draw precision (20-29%) is about the draw base rate
(26%), so draws are not predictable with these features.

## Known limits (stated in the README)
760 test matches cannot detect log-loss differences below about 0.015; the ablation uses fixed hyper-parameters; the bookmaker
odds may be closing odds; no line-up or injury information.
