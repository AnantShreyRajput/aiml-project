# ⚽ Premier League Match Predictor

An AI/ML project that predicts the result of a Premier League match — **Home win, Draw or Away win** — from
team form, Elo ratings, head-to-head records and squad strength. It covers the full machine-learning pipeline:
data collection → cleaning → feature engineering → model training → evaluation → a web app.

**Result:** a Random Forest that is right **50.5%** of the time on 760 unseen matches (2024-25 and 2025-26).
That beats the "always pick the home team" baseline (41.7%) and comes within about 1 point of professional
bookmakers (51.6%).

---

## 1. Project structure

```
aiml project/
├── app.py                      ← Streamlit web app (Step 6)
├── run_app.bat                 ← double-click to start the app on Windows
├── requirements.txt            ← Python packages needed
├── README.md                   ← this file
├── data/
│   ├── raw/                    ← Step 1: downloaded data
│   │   ├── matches/            match results, shots, corners, cards, odds, Elo
│   │   ├── players/            FPL player stats per season
│   │   └── squads/             team id → name lists
│   └── processed/              ← Steps 2–3: cleaned data and features
│       ├── matches_clean.csv
│       ├── team_matches_long.csv
│       ├── squad_strength_by_season.csv
│       ├── model_dataset_v1.csv
│       ├── model_features.csv  ← final table used for training
│       └── feature_list.csv
├── src/
│   ├── 02_clean_merge.py       ← Step 2
│   ├── 03_features.py          ← Step 3
│   ├── 04_train_models.py      ← Step 4
│   ├── 05_evaluate.py          ← Step 5
│   └── predict.py              ← builds features for any fixture (used by the app)
├── models/
│   ├── best_model.pkl          ← final model used by the app
│   └── best_model_dev_only.pkl ← model used for the test scores
└── results/
    ├── model_comparison.csv
    ├── evaluation_report.txt
    ├── best_params.json, feature_importance.csv, calibration_table.csv, accuracy_by_confidence.csv
    └── figures/                ← 6 charts for the report
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
Run these in order from the terminal, inside the project folder:
```
python src/02_clean_merge.py     # Step 2: clean + merge  -> data/processed/
python src/03_features.py        # Step 3: features       -> data/processed/model_features.csv
python src/04_train_models.py    # Step 4: train + tune   -> models/, results/model_comparison.csv  (~1 min)
python src/05_evaluate.py        # Step 5: evaluate       -> results/figures/, results/evaluation_report.txt
```

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

### Step 1 — Data collection
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

### Step 3 — Feature engineering (`src/03_features.py`)
111 features, all calculated **only from matches played before** the match being predicted:
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
Test results on 760 unseen matches:

| Model | Accuracy | Log loss (lower = better) |
|---|---|---|
| Bookmakers (reference) | 51.6% | 0.995 |
| **Random Forest (champion)** | **50.5%** | 1.016 |
| Elo difference only | 50.1% | 1.013 |
| Logistic Regression | 50.0% | 1.015 |
| Ensemble (average of 3) | 49.9% | 1.015 |
| XGBoost | 49.5% | 1.021 |
| Baseline: always home win | 41.7% | 1.083 |

Key findings (charts in `results/figures/`):
- **Elo difference** is by far the most important feature, followed by season goal difference and points per game.
- **Probabilities are well calibrated**: when the model says 60%, that result happens about 60% of the time.
- **Confidence matters**: predictions made with 70%+ confidence were right 69% of the time.
- **Draws are never predicted.** They are 26% of matches but almost never the single most likely outcome — bookmakers have the same problem.
- **Home advantage** dropped sharply in 2020-21, when matches were played without crowds.

### Step 6 — Web app (`app.py`)
A Streamlit app that builds the 27 features for any fixture from the latest data
(`src/predict.py`) and shows the prediction with an explanation. The live features were checked
against 150 historical matches and match the training features exactly.

---

## 5. Limitations and future work
- The model never predicts a draw; a draw threshold for evenly matched teams could improve this.
- No match-day information: injuries, suspensions, line-ups, manager changes.
- No possession or team-level xG for all seasons (those sources were not accessible).
- Football is highly random: even bookmakers get only about half of all results right.
- Ideas: add xG for all seasons, player-level line-up strength, a Poisson model to predict exact scores, and automatic data updates each week.

---

## 6. Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Altair · Streamlit · joblib
