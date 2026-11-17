"""Data audit + leakage checks. Run from project root AFTER steps 2 and 3:  python src/audit_data.py
Writes results/data_audit.txt (paste its numbers into the report) and exits with an error if a check fails.
"""
import os, sys
import numpy as np, pandas as pd
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
P, R = 'data/processed', 'results'
lines, fails = [], []
def say(s=''):
    print(s); lines.append(s)
def check(name, ok, detail=''):
    say(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok: fails.append(name)
 
raw = pd.read_csv('data/raw/matches/epl_matches_2016-17_to_2026-27.csv', parse_dates=['MatchDate'])
mc = pd.read_csv(f'{P}/matches_clean.csv', parse_dates=['MatchDate'])
mf = pd.read_csv(f'{P}/model_features.csv', parse_dates=['MatchDate'])
feats = pd.read_csv(f'{P}/feature_list.csv').feature.tolist()
 
# ---------------------------------------------------------------- 1. reconciliation of counts and periods
say('1. DATASET RECONCILIATION')
tab = pd.DataFrame({'raw_rows': raw.groupby('Season').size(), 'first_match': raw.groupby('Season').MatchDate.min().dt.date,
                    'last_match': raw.groupby('Season').MatchDate.max().dt.date, 'in_model_table': mf.groupby('Season').size()}).fillna(0)
tab['in_model_table'] = tab.in_model_table.astype(int)
tab['role'] = ['history only (no earlier season for squad features)'] + ['development (train + CV)'] * 7 + ['TEST'] * 2 + ['live / part-played: never trained or tested on']
say(tab.to_string())
dev, test, live = (mf.Season <= '2023-24').sum(), mf.Season.isin(['2024-25', '2025-26']).sum(), (mf.Season == '2026-27').sum()
say(f'raw {len(raw)} = 2016-17 history {int((raw.Season == "2016-17").sum())} + modelled {len(mf)}')
say(f'modelled {len(mf)} = development {dev} (2017-18..2023-24) + test {test} (2024-25, 2025-26) + live {live} (2026-27, excluded)')
say(f'complete seasons used for modelling: 9 x 380 = {9 * 380}; walk-forward evaluation seasons 2021-22..2025-26 = {5 * 380} matches')
check('every complete season has 380 matches', all(tab.loc[s, 'raw_rows'] == 380 for s in tab.index if s != '2026-27'))
 
# ---------------------------------------------------------------- 2. completed matches only
say('\n2. COMPLETED MATCHES ONLY')
check('no missing results in the model table', mf.Result.notna().all())
check('no match dated in the future', (mf.MatchDate <= pd.Timestamp.today()).all(), f'(latest match {mf.MatchDate.max():%Y-%m-%d})')
check('no duplicate fixtures', not mc.duplicated(['MatchDate', 'HomeTeam', 'AwayTeam']).any())
 
# ---------------------------------------------------------------- 3. Elo is a pre-match rating
say('\n3. IS ELO A PRE-MATCH RATING?')
lg = pd.read_csv(f'{P}/team_matches_long.csv', parse_dates=['Date']).sort_values(['Team', 'Date', 'MatchID'])
lg['dElo'] = lg.Elo - lg.groupby('Team').Elo.shift(1)
c_prev = lg.dElo.corr(lg.groupby('Team').Points.shift(1)); c_now = lg.dElo.corr(lg.Points)
say(f'corr(change in Elo going INTO a match, points of the PREVIOUS match) = {c_prev:.3f}   (should be clearly positive)')
say(f'corr(change in Elo going INTO a match, points of THIS match)        = {c_now:.3f}   (should be close to 0)')
check('Elo already reflects the previous match but not the current one', c_prev > 0.3 and abs(c_now) < 0.1)
 
# ---------------------------------------------------------------- 4. predictors are known before kick-off
# the model table must carry exactly the cleaned pre-match Elo for EVERY match (not just a sample)
chk = mf.merge(mc[['MatchID', 'HomeElo', 'AwayElo']], on='MatchID', suffixes=('', '_clean'))
elo_gap = max((chk.HomeElo - chk.HomeElo_clean).abs().max(), (chk.AwayElo - chk.AwayElo_clean).abs().max())
check('model_features Elo equals matches_clean Elo for all matches', elo_gap < 1e-9, f'(max gap {elo_gap:.2e}, {len(chk)} matches)')

say('\n4. EVERY PREDICTOR IS AVAILABLE BEFORE KICK-OFF')
post_match = ['FTHome', 'FTAway', 'FTResult', 'HTHome', 'HTAway', 'HTResult', 'PointsHome', 'PointsAway', 'Result'] + \
             [f'{s}{c}' for s in ('Home', 'Away') for c in ('Shots', 'Target', 'Fouls', 'Corners', 'Yellow', 'Red')]
bad = [c for c in feats if c in post_match or c.startswith(('Odd', 'Imp', 'Max', 'Over', 'Under', 'Handi', 'FT', 'HT'))]
check('no same-match statistic, result or bookmaker column is a feature', not bad, str(bad))
groups = {'Elo (pre-match rating)': [c for c in feats if 'Elo' in c],
          'Rolling form / streaks / season-to-date (shifted by one match)': [c for c in feats if '_L' in c or 'Streak' in c or 'Run' in c or 'Season' in c or 'Rest' in c or c.startswith('Attack_vs')],
          'Head-to-head (earlier meetings only)': [c for c in feats if 'H2H' in c],
          'FPL squad strength (PREVIOUS season)': [c for c in feats if 'prev_' in c or 'promoted' in c]}
for g, cols in groups.items(): say(f'   {len(cols):3d} features  {g}')
 
# Rebuild the features of a sample of matches the way the app does (src/predict.py): only matches dated BEFORE the kick-off
# are visible, and the form / head-to-head / difference logic is the same code as training (src/features.py).
# This proves the stored features contain nothing from the match itself or later, for ALL 27 model features.
say('\n   Re-computing all model features with src/predict.py from strictly earlier matches only (400 matches)...')
from predict import Predictor
pr = Predictor()
samp = mf[mf.Season >= '2018-19'].sample(400, random_state=1)
cols = list(pr.features)
err = {c: 0.0 for c in cols}
for _, r in samp.iterrows():
    X, _ = pr.features_for(r.HomeTeam, r.AwayTeam, r.MatchDate, home_elo=r.HomeElo, away_elo=r.AwayElo)
    X = X.fillna(pr.fill).iloc[0]
    for c in cols:
        err[c] = max(err[c], abs(float(X[c]) - float(r[c])))
for c, e in sorted(err.items(), key=lambda kv: -kv[1])[:5]: say(f'   largest gaps: {c:28s} {e:.2e}')
say(f'   max |recomputed - stored| over all {len(cols)} features and {len(samp)} matches: {max(err.values()):.2e}')
check('stored features equal features recomputed from earlier matches only', max(err.values()) < 1e-6)

# ---------------------------------------------------------------- 5. FPL squad features come from the previous season
say('\n5. FPL SQUAD FEATURES USE THE PREVIOUS SEASON')
sq = pd.read_csv(f'{P}/squad_strength_by_season.csv')
ds = pd.read_csv(f'{P}/model_dataset_v1.csv')
ex = ds[(ds.Season == '2024-25') & (ds.HomeTeam == 'Arsenal')].iloc[0]
ref = sq[(sq.Season == '2023-24') & (sq.Team == 'Arsenal')].squad_total_cost.iloc[0]
check('Arsenal 2024-25 matches carry Arsenal 2023-24 squad value', abs(ex.Home_prev_squad_total_cost - ref) < 1e-9, f'({ref:.1f}m)')
check('promoted teams are flagged (3 per season on average)', 1.5 < (ds.groupby('Season').Home_promoted.sum() / 19 * 1).mean() < 5)
 
say('\nRESULT: ' + ('ALL CHECKS PASSED' if not fails else 'FAILED -> ' + '; '.join(fails)))
os.makedirs(R, exist_ok=True); open(f'{R}/data_audit.txt', 'w').write('\n'.join(lines))
sys.exit(1 if fails else 0)
