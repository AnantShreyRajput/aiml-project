"""Single source of truth for the model's feature logic.

Used by src/03_features.py (all historical matches at once) and by src/predict.py (one future fixture), so the
training features and the live features can never drift apart. Every feature only looks at matches played BEFORE
the match it describes (rolling windows are shifted by one match).
"""
import numpy as np
import pandas as pd

H2H_DEFAULT = 1.35        # neutral head-to-head points per game when two teams have never met
DEV_END = '2023-24'       # last development season: gap-filling medians use development seasons only

BASE_STATS = ['Points', 'GF', 'GA', 'GD', 'Shots', 'ShotsAgainst', 'SoT', 'SoTAgainst', 'Corners', 'Fouls',
              'Yellow', 'Red', 'Win', 'CleanSheet']
FORM_DIFF = ['Points_L5', 'GF_L5', 'GA_L5', 'GD_L5', 'Shots_L5', 'SoT_L5', 'Win_L5', 'Points_L10', 'SeasonPPG',
             'SeasonGDpg', 'RestDays', 'Points_L3']
SQUAD = ['xi_avg_cost', 'xi_avg_points', 'xi_avg_ict', 'squad_total_cost', 'team_goals', 'gk_pts', 'def_pts',
         'mid_pts', 'att_pts']


def _streak(s, val):
    out, run = [], 0
    for x in s.shift(1).fillna(-1):
        run = run + 1 if x == val else 0
        out.append(run)
    return out


def team_form_features(lg):
    """Form features for every row of a long table (one row per team per match).

    `lg` needs: MatchID, Date, Season, Team, IsHome and the BASE_STATS source columns (GF, GA, Points, Shots, ...).
    Rows whose own results are NaN (a fixture still to be played) are allowed: their features only use earlier rows.
    Returns a DataFrame indexed by (MatchID, Team)."""
    lg = lg.sort_values(['Team', 'Date', 'MatchID']).copy()
    lg['Win'] = (lg.Points == 3).astype(int)
    lg['CleanSheet'] = (lg.GA == 0).astype(int)
    lg['GD'] = lg.GF - lg.GA
    g = lg.groupby('Team')
    feats = pd.DataFrame({'MatchID': lg.MatchID, 'Team': lg.Team})
    # --- rolling form: mean of previous N matches (shift(1) removes the current match) ---
    for N in (3, 5, 10):
        for c in BASE_STATS:
            if N != 5 and c not in ('Points', 'GF', 'GA'):
                continue
            feats[f'{c}_L{N}'] = g[c].transform(lambda s: s.shift(1).rolling(N, min_periods=1).mean())
    # --- venue form: last 5 at THIS venue (home team -> its home games, away team -> its away games) ---
    for c in ('Points', 'GF', 'GA'):
        feats[f'{c}_venue_L5'] = lg.groupby(['Team', 'IsHome'])[c].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    # --- streaks / momentum ---
    feats['WinStreak'] = g['Win'].transform(lambda s: _streak(s, 1))
    feats['UnbeatenRun'] = g['Points'].transform(lambda s: pd.Series(_streak((s > 0).astype(int), 1), index=s.index))
    feats['LossStreak'] = g['Points'].transform(lambda s: pd.Series(_streak((s == 0).astype(int), 1), index=s.index))
    # --- season-to-date table (before this match) ---
    lg['SeasonPts'] = lg.groupby(['Team', 'Season']).Points.transform(lambda s: s.shift(1).cumsum()).fillna(0)
    lg['SeasonGP'] = lg.groupby(['Team', 'Season']).cumcount()
    lg['SeasonGD'] = lg.groupby(['Team', 'Season']).GD.transform(lambda s: s.shift(1).cumsum()).fillna(0)
    feats['SeasonPPG'] = (lg.SeasonPts / lg.SeasonGP.replace(0, np.nan)).values
    feats['SeasonGDpg'] = (lg.SeasonGD / lg.SeasonGP.replace(0, np.nan)).values
    feats['SeasonGP'] = lg.SeasonGP.values
    # --- rest days since previous match ---
    feats['RestDays'] = g['Date'].transform(lambda s: s.diff().dt.days).clip(upper=14).values
    return feats.set_index(['MatchID', 'Team'])


def h2h_ppg(past, home):
    """Home team's points per game over its last 5 meetings with this opponent (either venue).
    `past` = [(home_team_of_that_match, home_points_of_that_match), ...] in date order, earlier meetings only.
    Returns NaN when the teams have never met."""
    pts = [p if t == home else {3: 0, 1: 1, 0: 3}[p] for t, p in past[-5:]]
    return float(np.mean(pts)) if pts else np.nan


def fill_season_start(m):
    """Season openers have no games played: points / goal difference per game are 0. Done BEFORE taking home-minus-away
    differences, so a team with 0 games vs a team with 1 game gets the real gap, not a missing value."""
    for c in [c for c in m.columns if 'SeasonPPG' in c or 'SeasonGDpg' in c]:
        m[c] = m[c].fillna(0)
    return m


def add_diff_features(m):
    """Home-minus-away difference features from the Home_* / Away_* columns of a match table (in place)."""
    m['Elo_diff'] = m.HomeElo - m.AwayElo
    for c in FORM_DIFF:
        m[f'{c}_diff'] = m[f'Home_{c}'] - m[f'Away_{c}']
    for c in SQUAD:
        m[f'prev_{c}_diff'] = m[f'Home_prev_{c}'] - m[f'Away_prev_{c}']
    m['Attack_vs_Defence_home'] = m.Home_GF_L5 - m.Away_GA_L5   # home attack against away defence
    m['Attack_vs_Defence_away'] = m.Away_GF_L5 - m.Home_GA_L5
    return m


def squad_floor(prev_squad, cols=SQUAD):
    """Squad strength given to a newly promoted team (no previous EPL season): the average of the 3 weakest teams of the
    previous season, per column. `prev_squad` = squad rows of the previous season; `cols` = the columns to fill."""
    return prev_squad[cols].apply(lambda c: c.nsmallest(3).mean())
