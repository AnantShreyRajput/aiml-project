"""Step 2: clean + merge raw EPL data -> data/processed/. Run from project root."""
import pandas as pd, numpy as np, glob, os
RAW, OUT = 'data/raw', 'data/processed'
os.makedirs(OUT, exist_ok=True)
SEASONS = ['2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25','2025-26','2026-27']

# canonical names = football-data names used in the match file
NAME_MAP = {'Man Utd':'Man United','Spurs':"Tottenham","Nottm Forest":"Nott'm Forest",
            "Nott'm Forest":"Nott'm Forest",'Sheffield Utd':'Sheffield United','Leeds':'Leeds',
            'Newcastle':'Newcastle','Wolves':'Wolves','West Brom':'West Brom','Luton':'Luton'}
canon = lambda n: NAME_MAP.get(n, n)

# ---------- 1. matches ----------
m = pd.read_csv(f'{RAW}/matches/epl_matches_2016-17_to_2026-27.csv')
m = m.drop(columns=[c for c in m.columns if c.startswith('C_')] + ['MatchTime','Division'])
m['MatchDate'] = pd.to_datetime(m['MatchDate'])
for c in ('HomeTeam','AwayTeam'): m[c] = m[c].map(canon)
m = m.drop_duplicates(['MatchDate','HomeTeam','AwayTeam']).sort_values('MatchDate').reset_index(drop=True)
m['Result'] = m['FTResult'].map({'H':0,'D':1,'A':2})          # target
# implied bookmaker probabilities (normalised, removes overround)
inv = 1/m[['OddHome','OddDraw','OddAway']]
imp = inv.div(inv.sum(axis=1), axis=0); imp.columns = ['ImpHome','ImpDraw','ImpAway']
m = pd.concat([m, imp], axis=1)
# missing Elo -> previous known Elo of that team
for side in ('Home','Away'):
    pass
m['HomeElo'] = m['HomeElo'].fillna(m.groupby('HomeTeam')['HomeElo'].transform('mean'))
m['AwayElo'] = m['AwayElo'].fillna(m.groupby('AwayTeam')['AwayElo'].transform('mean'))
m['HandiSize'] = m['HandiSize'].fillna(0)
m['MatchID'] = np.arange(len(m))
m['PointsHome'] = m['FTResult'].map({'H':3,'D':1,'A':0}); m['PointsAway'] = m['FTResult'].map({'H':0,'D':1,'A':3})
assert m[['Result','FTHome','HomeShots','OddHome']].isna().sum().sum() == 0
m.to_csv(f'{OUT}/matches_clean.csv', index=False)

# ---------- 2. long format (one row per team per match) - used for rolling form in step 3 ----------
def side(s, o):
    d = pd.DataFrame({'MatchID':m.MatchID,'Date':m.MatchDate,'Season':m.Season,'Team':m[f'{s}Team'],'Opponent':m[f'{o}Team'],
        'IsHome':int(s=='Home'),'GF':m[f'FT{s}'],'GA':m[f'FT{o}'],'Shots':m[f'{s}Shots'],'ShotsAgainst':m[f'{o}Shots'],
        'SoT':m[f'{s}Target'],'SoTAgainst':m[f'{o}Target'],'Corners':m[f'{s}Corners'],'Fouls':m[f'{s}Fouls'],
        'Yellow':m[f'{s}Yellow'],'Red':m[f'{s}Red'],'Points':m[f'Points{s}'],'Elo':m[f'{s}Elo']})
    return d
long = pd.concat([side('Home','Away'), side('Away','Home')]).sort_values(['Date','MatchID']).reset_index(drop=True)
long.to_csv(f'{OUT}/team_matches_long.csv', index=False)

# ---------- 3. squad strength per team-season (from FPL player data) ----------
tl = pd.read_csv(f'{RAW}/squads/master_team_list.csv')
tl['team_name'] = tl['team_name'].map(canon)
# master list stops at 2023-24 -> fall back to each season's own teams_<season>.csv
extra = []
for s in SEASONS:
    if s not in set(tl.season):
        t = pd.read_csv(f'{RAW}/squads/teams_{s}.csv')
        extra.append(pd.DataFrame({'season':s,'team':t['id'],'team_name':t['name'].map(canon)}))
tl = pd.concat([tl]+extra, ignore_index=True)
rows = []
for s in SEASONS:
    f = f'{RAW}/players/players_raw_{s}.csv'
    p = pd.read_csv(f)
    p['team_name'] = p['team'].map(dict(zip(tl[tl.season==s].team, tl[tl.season==s].team_name)))
    p = p.dropna(subset=['team_name']); p['season'] = s
    for c in ('expected_goals','expected_assists','ict_index','form','now_cost','minutes','total_points'):
        p[c] = pd.to_numeric(p.get(c, np.nan), errors='coerce')
    p['now_cost'] = p['now_cost']/10
    for team, g in p.groupby('team_name'):
        top = g.nlargest(11, 'minutes')           # the 11 players who played most = the regular XI
        pos = lambda t: g[g.element_type==t].nlargest({1:1,2:4,3:4,4:2}[t],'minutes')
        rows.append(dict(Season=s, Team=team,
            xi_avg_cost=top.now_cost.mean(), xi_avg_points=top.total_points.mean(), xi_avg_ict=top.ict_index.mean(),
            squad_total_cost=g.now_cost.sum(), team_goals=g.goals_scored.sum(), team_assists=g.assists.sum(),
            team_xg=g.expected_goals.sum() if g.expected_goals.notna().any() else np.nan,
            gk_pts=pos(1).total_points.mean(), def_pts=pos(2).total_points.mean(),
            mid_pts=pos(3).total_points.mean(), att_pts=pos(4).total_points.mean(),
            injured_now=int((g.status.isin(['i','d','s','u']) & (g.minutes>500)).sum())))
sq = pd.DataFrame(rows)
sq.to_csv(f'{OUT}/squad_strength_by_season.csv', index=False)

# prior-season strength joined to each match (end-of-season stats of the SAME season would leak the future)
prev = {s: SEASONS[i-1] for i, s in enumerate(SEASONS) if i}
feat = [c for c in sq.columns if c not in ('Season','Team','injured_now','team_xg')]  # team_xg only exists from 2022-23, too sparse
p = sq.copy(); p['Season'] = p['Season'].map({v:k for k,v in prev.items()})   # shift forward one season
p = p.dropna(subset=['Season'])
for side_ in ('Home','Away'):
    q = p[['Season','Team']+feat].rename(columns={'Team':f'{side_}Team', **{c:f'{side_}_prev_{c}' for c in feat}})
    m = m.merge(q, on=['Season',f'{side_}Team'], how='left')
# promoted teams have no prior-season EPL record: flag + fill with the weakest-3 average that season
for side_ in ('Home','Away'):
    cols = [f'{side_}_prev_{c}' for c in feat]
    m[f'{side_}_promoted'] = m[cols[0]].isna().astype(int)
for c in feat:
    for side_ in ('Home','Away'):
        col = f'{side_}_prev_{c}'
        floor = m['Season'].map(p.groupby('Season')[c].apply(lambda x: x.nsmallest(3).mean()))  # 3 weakest distinct teams
        m[col] = m[col].fillna(floor)
m = m[m.Season != '2016-17'].reset_index(drop=True)      # first season has no "previous season" squad data
m.to_csv(f'{OUT}/model_dataset_v1.csv', index=False)
print('matches_clean', len(pd.read_csv(f'{OUT}/matches_clean.csv')), '| long', len(long), '| squad rows', len(sq), '| model_dataset_v1', m.shape)
print('NaNs left:', m.isna().sum()[lambda s: s>0].to_dict())
