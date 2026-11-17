"""Step 3: feature engineering. Every feature uses ONLY matches played before the match date. Run from project root."""
import os, sys
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import team_form_features, h2h_ppg, fill_season_start, add_diff_features, H2H_DEFAULT, DEV_END
P='data/processed'
m  = pd.read_csv(f'{P}/model_dataset_v1.csv', parse_dates=['MatchDate'])
lg = pd.read_csv(f'{P}/team_matches_long.csv', parse_dates=['Date'])

# all form logic (rolling windows, venue form, streaks, season-to-date, rest days) lives in src/features.py,
# the same code src/predict.py uses for live fixtures
feats=team_form_features(lg)

def attach(m,side):
    f=feats.copy(); f.columns=[f'{side}_{c}' for c in f.columns]
    return m.merge(f.reset_index().rename(columns={'Team':f'{side}Team'}),on=['MatchID',f'{side}Team'],how='left')
m=attach(m,'Home'); m=attach(m,'Away')

# --- head-to-head: home team's points per game in the last 5 meetings (either venue) ---
# uses ALL cleaned matches (incl. 2016-17) as history
allm=pd.read_csv(f'{P}/matches_clean.csv',parse_dates=['MatchDate']).sort_values(['MatchDate','MatchID'])
h2h=[];hist={}
for _,r in allm.iterrows():
    key=tuple(sorted((r.HomeTeam,r.AwayTeam))); past=hist.get(key,[])[-5:]
    h2h.append((r.MatchID,h2h_ppg(past,r.HomeTeam),len(past)))
    hist.setdefault(key,[]).append((r.HomeTeam,r.PointsHome))
m=m.merge(pd.DataFrame(h2h,columns=['MatchID','H2H_HomePPG','H2H_n']),on='MatchID')

# --- difference features (home minus away): models learn these best ---
# Season-opener gaps (0 games played) are filled with 0 BEFORE taking home-minus-away differences,
# so a team with 0 games vs a team with 1 game gets the real gap, not a missing value.
m=fill_season_start(m)
m=add_diff_features(m)
m['Month']=m.MatchDate.dt.month

# --- fill remaining gaps from the very start of the data ---
m['H2H_HomePPG']=m.H2H_HomePPG.fillna(H2H_DEFAULT)     # neutral = average points when no history
m=m.sort_values('MatchDate').reset_index(drop=True)
rolling=[c for c in m.columns if '_L' in c or 'Streak' in c or 'Run' in c or 'RestDays' in c or 'venue' in c or c.startswith('Attack_vs')]

# Remaining gaps are filled with the median of the DEVELOPMENT seasons only (<= 2023-24), never the test seasons.
dev_median = m.loc[m.Season <= DEV_END, rolling].median()
print('values filled with the dev-season median:', int(m[rolling].isna().sum().sum()))
m[rolling] = m[rolling].fillna(dev_median)
# --- final files ---
ID=['MatchID','MatchDate','Season','HomeTeam','AwayTeam']
TARGET=['Result','FTResult','FTHome','FTAway']
ODDS=['OddHome','OddDraw','OddAway','ImpHome','ImpDraw','ImpAway']   # kept ONLY to compare against bookmakers
leak=['FTHome','FTAway','FTResult','HTHome','HTAway','HTResult','PointsHome','PointsAway']+[f'{s}{c}' for s in('Home','Away') for c in('Shots','Target','Fouls','Corners','Yellow','Red')]
drop=set(leak+['Result','OddHome','OddDraw','OddAway','MaxHome','MaxDraw','MaxAway','Over25','Under25','MaxOver25','MaxUnder25','HandiSize','HandiHome','HandiAway','ImpHome','ImpDraw','ImpAway']+ID)
X=[c for c in m.columns if c not in drop and not c.startswith(('Form3','Form5'))]
out=m[ID+X+['Result']+ODDS]
out.to_csv(f'{P}/model_features.csv',index=False)
pd.Series(X,name='feature').to_csv(f'{P}/feature_list.csv',index=False)
print('rows',len(out),'features',len(X)); print('NaNs:',int(out[X].isna().sum().sum()))
