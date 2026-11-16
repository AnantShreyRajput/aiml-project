"""Step 6: feature ablation + football-specific baselines, on IDENTICAL season splits.
Run from project root:  python src/06_ablation.py      (takes ~6 minutes, mostly the Dixon-Coles refits)
 
Walk-forward (expanding window): for each season S in 2021-22 ... 2025-26 the models are trained on all
seasons from 2017-18 up to S-1 and predict S. Every model sees exactly the same train/test rows.
Output: results/oof_predictions.csv  (one row per match per model: pH, pD, pA)
        Metrics are computed from this file in step 7 (src/07_probability_eval.py).
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dixon_coles import walk_forward
 
P = 'data/processed'
EVAL_SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']   # 2024-25 + 2025-26 = the held-out test seasons
 
f = pd.read_csv(f'{P}/model_features.csv', parse_dates=['MatchDate'])
f = f[f.Season != '2026-27'].reset_index(drop=True)            # 2026-27 is a part-played season: never used for evaluation
ALL = pd.read_csv(f'{P}/feature_list.csv').feature.tolist()
 
# ---------------- feature groups (every column is known before kick-off) ----------------
ELO = ['HomeElo', 'AwayElo', 'Elo_diff']
FORM = ['H2H_HomePPG'] + [c for c in ALL if c.endswith('_diff') and not c.startswith('prev_') and c != 'Elo_diff']
FPL = [c for c in ALL if c.startswith('prev_') and c.endswith('_diff')] + ['Home_promoted', 'Away_promoted']
SETS = {
    'A Elo only': ['Elo_diff'],                  # the "Elo-only" baseline of the proposal
    'B Elo+Form': ELO + FORM,
    'C Elo+Form+FPL': ELO + FORM + FPL,          # = the 27 "compact" features of the original project
    'D Elo+FPL': ELO + FPL,                      # extra: isolates what FPL adds WITHOUT form
}
assert len(SETS['C Elo+Form+FPL']) == 27, len(SETS['C Elo+Form+FPL'])
assert not any(c.startswith(('Odd', 'Imp')) for s in SETS.values() for c in s), 'bookmaker odds must never be a feature'
 
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
RF = lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=25, n_jobs=2, random_state=42)
 
rows = []
def add(season, name, d, P_):
    out = d[['Season', 'MatchID', 'Result']].copy()
    out['model'] = name
    out[['pH', 'pD', 'pA']] = np.asarray(P_)
    rows.append(out)
 
# ---------------- Dixon-Coles (cached, because it is the slow part) ----------------
DC_FILE = 'results/dc_predictions.csv'
if os.path.exists(DC_FILE):
    dc = pd.read_csv(DC_FILE)
else:
    mc = pd.read_csv(f'{P}/matches_clean.csv', parse_dates=['MatchDate']).set_index('MatchID', drop=False)
    parts = []
    for s in EVAL_SEASONS:
        ids = f.loc[f.Season == s, 'MatchID'].values
        print('Dixon-Coles walk-forward', s, flush=True)
        pr = walk_forward(mc, ids)
        pr['MatchID'] = ids
        parts.append(pr)
    dc = pd.concat(parts)
    dc.to_csv(DC_FILE, index=False)
dc = dc.set_index('MatchID')
 
# ---------------- all other models ----------------
for s in EVAL_SEASONS:
    tr, te = f[f.Season < s], f[f.Season == s]
    print('season', s, '| train', len(tr), '| test', len(te), flush=True)
    add(s, 'Baseline: always home win', te, DummyClassifier(strategy='prior').fit(tr[ALL], tr.Result).predict_proba(te[ALL]))
    add(s, 'Dixon-Coles (Poisson)', te, dc.loc[te.MatchID, ['pH', 'pD', 'pA']].values)
    add(s, 'Bookmakers (reference)', te, te[['ImpHome', 'ImpDraw', 'ImpAway']].values)
    for sname, cols in SETS.items():
        add(s, f'LR | {sname}', te, LR().fit(tr[cols], tr.Result).predict_proba(te[cols]))
        add(s, f'RF | {sname}', te, RF().fit(tr[cols], tr.Result).predict_proba(te[cols]))
 
oof = pd.concat(rows, ignore_index=True)
oof.to_csv('results/oof_predictions.csv', index=False)
print('saved results/oof_predictions.csv', oof.shape, '| models:', oof.model.nunique())
