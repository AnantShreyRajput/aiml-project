# ⚽ Premier League Match Predictor

An AI/ML project that predicts the result of a Premier League match — **Home win, Draw or Away win** — from
team form, Elo ratings, head-to-head records and squad strength. It covers the full machine-learning pipeline:
data collection → cleaning → feature engineering → model training → evaluation → a web app.

**Result:** a reproducible comparative study of pre-match EPL prediction, not a new prediction algorithm. On 760 unseen matches
(2024-25 and 2025-26) the selected Random Forest is right 50.7% of the time (always-home-win: 41.7%). A feature ablation on five
walk-forward seasons shows that previous-season FPL squad features do **not** consistently improve log loss over Elo or
Elo + rolling form, and a Dixon-Coles goal model performs similarly. Bookmakers (51.6% accuracy, log loss 0.995) remain
significantly better. See `REVISION_NOTES.md` for how this answers the faculty review.

---

## 1. Project structure

```
aiml-project/
├── app.py                      ← Streamlit web app (Step 6)
├── run_app.bat                 ← double-click to start the app on Windows
├── requirements.txt            ← Python packages needed (minimum versions)
├── requirements-lock.txt       ← exact versions the reported results were produced with
├── README.md                   ← this file
├── REVISION_NOTES.md           ← response to the faculty review (5 required modifications)
├── data/
│   ├── raw/                    ← Step 1: downloaded data (manual download; MANIFEST.csv = checksums, README.md = sources)
│   │   ├── matches/            match results, shots, corners, cards, odds, Elo
│   │   ├── players/            FPL player stats per season
│   │   └── squads/             team id → name lists
│   └── processed/              ← Steps 2–3: cleaned data and features
│       ├── matches_clean.csv, team_matches_long.csv, squad_strength_by_season.csv
│       ├── model_dataset_v1.csv
│       ├── model_features.csv  ← final table used for training
│       └── feature_list.csv
├── src/
│   ├── 01_check_raw_data.py    ← Step 1: verifies the raw files against data/raw/MANIFEST.csv, says where to download them
│   ├── features.py             ← the feature logic (used by BOTH step 03 and predict.py, so they cannot drift apart)
│   ├── 02_clean_merge.py       ← Step 2
│   ├── 03_features.py          ← Step 3
│   ├── 04_train_models.py      ← Step 4
│   ├── 05_evaluate.py          ← Step 5
│   ├── audit_data.py           ← data reconciliation + leakage checks (after step 3)
│   ├── dixon_coles.py          ← Dixon-Coles goal model (baseline, imported by step 06)
│   ├── 06_ablation.py          ← Elo vs +form vs +FPL, walk-forward, with baselines
│   ├── 07_probability_eval.py  ← log loss, Brier, calibration, draw precision/recall, bootstrap CIs
│   └── predict.py              ← builds features for any fixture with the same code as step 03 (used by the app)
├── models/
│   ├── best_model.pkl          ← model used by the app (fitted on 2017-18 → 2025-26)
│   └── best_model_dev_only.pkl ← model behind the test scores (fitted on 2017-18 → 2023-24)
├── results/                    ← all tables, reports and the 8 figures in results/figures/
└── screenshots/                ← app screenshots for the report
```

---

## 2. Running the project in VS Code

### One-time setup
1. **Install Python 3.11 or newer** from <https://www.python.org/downloads/>.
   During installation, tick **"Add Python to PATH"**.
2. **Install VS Code** from <https://code.visualstudio.com/>, open it, go to the Extensions tab
   (`Ctrl+Shift+X`) and install the **Python** extension by Microsoft.
3. **Open the project:** `File → Open Folder…` → choose the `aiml project` folder on your Desktop.
4. **Open a terminal:** `Terminal → New Terminal` (or `` Ctrl+` ``).
5. **(Recommended) create a virtual environment**, so the project's packages stay separate:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   You should now see `(.venv)` at the start of the terminal line.
   If PowerShell says *"running scripts is disabled"*, run this once and try again:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
   VS Code may ask "use this environment for the workspace?" — click **Yes**.
6. **Install the packages:**
   ```
   pip install -r requirements.txt
   ```

### Run the app
```
streamlit run app.py
```
Your browser opens at <http://localhost:8501>. If it doesn't, open that address yourself.
To stop the app, click the terminal and press `Ctrl+C`.

Next time, you only need to open the folder in VS Code, run `.venv\Scripts\activate`, then `streamlit run app.py`.

### Re-run the full pipeline (optional)
Run these in order from the terminal, inside the project folder (about 10 minutes in total, most of it step 06 if the
Dixon-Coles cache `results/dc_predictions.csv` is deleted; with the cache it takes under a minute):
```
python src/01_check_raw_data.py    # Step 1: raw data present and unchanged (see data/raw/README.md for the sources)
python src/02_clean_merge.py       # Step 2: clean + merge  -> data/processed/
python src/03_features.py          # Step 3: features       -> data/processed/model_features.csv
python src/audit_data.py           # data audit + leakage checks (must print ALL CHECKS PASSED)
python src/04_train_models.py      # Step 4: train + tune   -> models/, results/model_comparison.csv  (~30 s)
python src/05_evaluate.py          # Step 5: evaluate       -> results/figures/, results/evaluation_report.txt
python src/06_ablation.py          # ablation + baselines   -> results/oof_predictions.csv
python src/07_probability_eval.py  # probability / draw evaluation -> results/*.csv, probability_report.txt
```
Step 1 (downloading the raw data) is manual, because the sources are public web pages rather than an API; `01_check_raw_data.py`
checks every raw file against the SHA-256 checksums in `data/raw/MANIFEST.csv` and prints where each file comes from if one is missing.
To add newer matches later, update the raw match file, run `python src/01_check_raw_data.py --write-manifest`, then run steps 02-04.
Exact numbers can differ in the 3rd decimal with other library versions; `requirements-lock.txt` lists the versions used for
every number in this README.

### Quick alternative (no VS Code)
Double-click **`run_app.bat`**. It installs the packages and starts the app.

### Troubleshooting
| Problem | Fix |
|---|---|
| `python` is not recognized | Python is not on PATH. Reinstall Python and tick "Add Python to PATH", or use `py` instead of `python`. |
| `streamlit` is not recognized | Use `python -m streamlit run app.py`. |
| `ModuleNotFoundError` | The packages aren't installed in the active environment: activate `.venv` and run `pip install -r requirements.txt` again. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Model file won't load (different scikit-learn version) | Nothing to do: the app automatically retrains the same model in a few seconds. |
| Different numbers when re-running the pipeline | Install the exact versions in `requirements-lock.txt` (`requirements.txt` only bounds them). |

---

## 3. Using the app

- **Predict a match** — choose a home team and an away team. The app shows:
  - the probability of a home win, draw and away win, with a bar chart;
  - the most likely result, and how often predictions at that confidence were right in testing;
  - a comparison table (Elo, last-5 form, goals, shots on target, season points, squad value);
  - each team's last 5 results and their last 5 head-to-head meetings.
- **Model performance** — the model comparison table and all evaluation charts.
- **How it works** — a short explanation of the inputs, the model and its limits.

---

## 4. The pipeline, step by step

### Step 1 — Data collection (`src/01_check_raw_data.py`)
| Data | Source | Coverage |
|---|---|---|
| Match results, shots, shots on target, corners, fouls, cards, betting odds, Elo | football-data.co.uk (via GitHub mirror `xgabora/Club-Football-Match-Data-2000-2025`) | 3,820 EPL matches, 2016-17 → 2026-27 (first 20 games) |
| Player stats (price, minutes, goals, assists, form, ICT index, injury status) | Official Fantasy Premier League API (via `vaastav/Fantasy-Premier-League`) | every player, every season |

### Step 2 — Cleaning and merging (`src/02_clean_merge.py`)
- Unified team names across sources (e.g. Man Utd → Man United, Spurs → Tottenham).
- Fixed season labels (the COVID-delayed 2019-20 season ended in July 2020).
- Target column `Result`: 0 = Home win, 1 = Draw, 2 = Away win.
- Built a **squad strength** table per team per season from FPL player data
  (average price, points and ICT index of the 11 most-used players, points by position, squad value).
- Each match gets the **previous season's** squad strength, because a season's end-of-year totals would reveal its own results.
- Promoted teams (no previous EPL season) are flagged and given the average of the 3 weakest teams.

### Step 3 — Feature engineering (`src/03_features.py`, logic in `src/features.py`)
111 features (the models use a compact set of 27), all calculated **only from matches played before** the match being predicted:
- **Recent form** over the last 3, 5 and 10 games: points, goals for/against, goal difference, shots, shots on target, wins.
- **Venue form**: home team's last 5 home games, away team's last 5 away games.
- **Momentum**: win streak, unbeaten run, loss streak, days of rest.
- **Season so far**: points per game and goal difference per game.
- **Strength**: Elo ratings, head-to-head record over the last 5 meetings, last season's squad strength.
- **Difference features** (home minus away), which models learn from most easily.

### Step 4 — Model training (`src/04_train_models.py`)
- **Time-based split, never random:** development = 2017-18 → 2023-24 (2,660 matches);
  test = 2024-25 + 2025-26 (760 matches), not touched until the end.
- **Season-by-season cross-validation:** train on earlier seasons, validate on 2021-22, 2022-23 and 2023-24.
- Models: Logistic Regression, Random Forest, XGBoost, each tried on all 111 features and on a compact set of 27.
- The champion is chosen by validation log loss **before** looking at the test set: **Random Forest**
  (300 trees, max depth 4, min 25 samples per leaf, 27 compact features).

### Step 5 — Evaluation (`src/05_evaluate.py`)
Test results on 760 unseen matches (models fitted on 2017-18 → 2023-24 only):

| Model | Accuracy | Log loss (lower = better) |
|---|---|---|
| Bookmakers (reference) | 51.6% | 0.995 |
| **Random Forest (selected by CV)** | **50.7%** | 1.016 |
| Elo difference only | 50.1% | 1.013 |
| Logistic Regression | 50.1% | 1.015 |
| Ensemble (average of 3) | 50.0% | 1.015 |
| XGBoost | 49.6% | 1.021 |
| Baseline: always home win | 41.7% | 1.083 |

The three tuned models are tied on cross-validation (log loss 0.9605 / 0.9608 / 0.9622) and on the test set the simple
Elo-only model is as good as any of them, so the Random Forest is *selected*, not shown to be better. Key findings
(charts in `results/figures/`):
- **Elo difference** is by far the most important feature, followed by season goal difference and points per game.
- **Probabilities are reasonably calibrated** (expected calibration error about 4 points). On the 760 test matches the most
  confident picks look over-confident (the 60-70% and 70%+ bands were right 58% and 70% of the time against 64% and 76% stated),
  but with only 120 and 79 matches that is within sampling noise: over the 1,900 walk-forward matches (2021-22 → 2025-26)
  the same bands are right 65% and 78% against 65% and 76% stated.
- **Confidence matters**: walk-forward accuracy rises from 39% (picks under 40% confidence) to 78% (70%+ band). The app shows these
  walk-forward hit rates, which use more matches and do not depend on the refitted production model.
- **Draws are never predicted.** They are 26% of matches but are almost never the single most likely outcome — bookmakers have the same problem.
- **Home advantage** dropped sharply in 2020-21, when matches were played without crowds.

### Step 6 — Web app (`app.py`)
A Streamlit app that builds the 27 features for any fixture from the latest data
(`src/predict.py`) and shows the prediction with an explanation. Live features are built with the same code as the training
features (`src/features.py`); `src/audit_data.py` checks that all 27 model features recomputed from earlier matches equal the stored ones
(also verified for all 3,440 matches: largest difference 6e-14). The deployed model is fitted on all 3,420 completed matches
(2017-18 → 2025-26); the performance tab reports the dev-only model's test scores, and the "right X% of the time" note uses
walk-forward hit rates (`results/accuracy_by_confidence_walkforward.csv`). If the match data is more than 30 days old the app shows a warning.

---

## 5. Limitations and future work
- **Draws are not predictable with these features.** The draw rule (predict a draw when home and away look evenly matched,
  threshold chosen on 2021-22 → 2023-24) reaches draw precision of 20-29% against a draw base rate of 26% on the test seasons,
  i.e. no skill beyond guessing, and it lowers overall accuracy from about 50% to 43-47%. The app therefore still shows the
  draw *probability* but never names a draw as the pick.
- The match data ends on 31 Aug 2026 (20 games of 2026-27); predictions are as of that date. The app warns when the data is old.
- No match-day information: injuries, suspensions, line-ups, manager changes.
- No possession or team-level xG for all seasons (those sources were not accessible).
- Bookmaker odds are used only as a reference. They may be closing odds, which already contain team news the model cannot see.
- 760 test matches are too few to detect log-loss differences smaller than about 0.015, so "no consistent improvement" means
  *no evidence of improvement*, not proof of none. The ablation models use fixed (not per-feature-set tuned) hyper-parameters.
- Dixon-Coles gives a newly promoted team average strength, which handicaps it slightly against the feature models.
- Football is highly random: even bookmakers get only about half of all results right.
- Ideas: add xG for all seasons, player-level line-up strength, per-set tuning in the ablation, and automatic data updates each week.

---
## 6. Review revisions (ablation, baselines, audit)
| Script | What it does | Run order |
|---|---|---|
| `src/audit_data.py` | Reconciles counts, proves no leakage (Elo for every match; all 27 model features rebuilt from earlier matches), checks only completed matches are used | after step 3 |
| `src/dixon_coles.py` | Time-decayed Dixon-Coles (Poisson) goal model used as a baseline | imported by step 06 |
| `src/06_ablation.py` | Elo-only vs +form vs +form+FPL on identical walk-forward seasons, plus baselines | after step 5 |
| `src/07_probability_eval.py` | Log loss, Brier, calibration, draw precision/recall, per-season and bootstrap uncertainty | after step 06 |

**Ablation headline (walk-forward, 2021-22 → 2025-26, mean log loss over 5 seasons):** Elo-only LR 0.981, Elo + form LR 0.979,
Elo + form + FPL LR 0.984, Elo + form + FPL RF 0.982, Dixon-Coles 0.982, bookmakers 0.960. Paired bootstrap 95% intervals for every
FPL-related difference include zero; only the bookmakers are significantly better (difference about −0.024, interval −0.031 to −0.017).
Two evaluation protocols appear in this project: step 05 fits on 2017-18 → 2023-24 and tests once on the two test seasons, while
steps 06-07 retrain every season on all earlier seasons (walk-forward), so the same model can have slightly different numbers in the two.

Data reconciliation: raw 3,820 matches = 380 (2016-17, history only) + 3,440 modelled. Modelled = 2,660 development
(2017-18..2023-24) + 760 test (2024-25, 2025-26) + 20 live 2026-27 games (never used for training or testing).
The deployed model is fitted on the 3,420 completed matches of 2017-18 → 2025-26.

---
## 7. Tech stack
Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · Matplotlib · Altair · Streamlit · joblib
