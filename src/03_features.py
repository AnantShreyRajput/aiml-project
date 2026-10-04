"""Step 3: feature engineering. Every feature uses ONLY matches played before the match date. Run from project root."""
import pandas as pd, numpy as np
P='data/processed'
m  = pd.read_csv(f'{P}/model_dataset_v1.csv', parse_dates=['MatchDate'])
lg = pd.read_csv(f'{P}/team_matches_long.csv', parse_dates=['Date']).sort_values(['Team','Date','MatchID'])

lg['Win']=(lg.Points==3).astype(int); lg['CleanSheet']=(lg.GA==0).astype(int); lg['GD']=lg.GF-lg.GA
base=['Points','GF','GA','GD','Shots','ShotsAgainst','SoT','SoTAgainst','Corners','Fouls','Yellow','Red','Win','CleanSheet']
g=lg.groupby('Team')
feats=pd.DataFrame({'MatchID':lg.MatchID,'Team':lg.Team})
# --- rolling form: mean of previous N matches (shift(1) removes the current match) ---
for N in (3,5,10):
    for c in base:
        if N!=5 and c not in ('Points','GF','GA'): continue
        feats[f'{c}_L{N}']=g[c].transform(lambda s:s.shift(1).rolling(N,min_periods=1).mean())
# --- venue form: last 5 at THIS venue (home team -> its home games, away team -> its away games) ---
for c in ('Points','GF','GA'):
    feats[f'{c}_venue_L5']=lg.groupby(['Team','IsHome'])[c].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
# --- streaks / momentum ---
def streak(s,val):
    out,run=[],0
    for x in s.shift(1).fillna(-1):
        run=run+1 if x==val else 0; out.append(run)
    return out
feats['WinStreak']=g['Win'].transform(lambda s:streak(s,1))
feats['UnbeatenRun']=g['Points'].transform(lambda s:pd.Series(streak((s>0).astype(int),1),index=s.index))
feats['LossStreak']=g['Points'].transform(lambda s:pd.Series(streak((s==0).astype(int),1),index=s.index))
# --- season-to-date table (before this match) ---
lg['SeasonPts']=lg.groupby(['Team','Season']).Points.transform(lambda s:s.shift(1).cumsum()).fillna(0)
lg['SeasonGP'] =lg.groupby(['Team','Season']).cumcount()
lg['SeasonGD'] =lg.groupby(['Team','Season']).GD.transform(lambda s:s.shift(1).cumsum()).fillna(0)
feats['SeasonPPG']=(lg.SeasonPts/lg.SeasonGP.replace(0,np.nan)).values
feats['SeasonGDpg']=(lg.SeasonGD/lg.SeasonGP.replace(0,np.nan)).values
feats['SeasonGP']=lg.SeasonGP.values
# --- rest days since previous match ---
feats['RestDays']=g['Date'].transform(lambda s:s.diff().dt.days).clip(upper=14).values
feats=feats.set_index(['MatchID','Team'])

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
    pts=[p if t==r.HomeTeam else {3:0,1:1,0:3}[p] for t,p in past]
    h2h.append((r.MatchID,np.mean(pts) if pts else np.nan,len(pts)))
    hist.setdefault(key,[]).append((r.HomeTeam,r.PointsHome))
m=m.merge(pd.DataFrame(h2h,columns=['MatchID','H2H_HomePPG','H2H_n']),on='MatchID')

# --- difference features (home minus away): models learn these best ---
m['Elo_diff']=m.HomeElo-m.AwayElo
for c in ['Points_L5','GF_L5','GA_L5','GD_L5','Shots_L5','SoT_L5','Win_L5','Points_L10','SeasonPPG','SeasonGDpg','RestDays','Points_L3']:
    m[f'{c}_diff']=m[f'Home_{c}']-m[f'Away_{c}']
for c in ['xi_avg_cost','xi_avg_points','xi_avg_ict','squad_total_cost','team_goals','gk_pts','def_pts','mid_pts','att_pts']:
    m[f'prev_{c}_diff']=m[f'Home_prev_{c}']-m[f'Away_prev_{c}']
m['Attack_vs_Defence_home']=m.Home_GF_L5-m.Away_GA_L5   # home attack against away defence
m['Attack_vs_Defence_away']=m.Away_GF_L5-m.Home_GA_L5
m['Month']=m.MatchDate.dt.month

# --- fill gaps from the very start of the data / season openers ---
m['H2H_HomePPG']=m.H2H_HomePPG.fillna(1.35)            # neutral = average points when no history
for c in [c for c in m.columns if ('SeasonPPG' in c or 'SeasonGDpg' in c)]: m[c]=m[c].fillna(0)
m=m.sort_values('MatchDate').reset_index(drop=True)
rolling=[c for c in m.columns if '_L' in c or 'Streak' in c or 'Run' in c or 'RestDays' in c or 'venue' in c or c.startswith('Attack_vs')]
m[rolling]=m[rolling].fillna(m[rolling].median())

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
