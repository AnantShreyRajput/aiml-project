"""Build the model's input features for ANY fixture, using only matches played before a given date.
Same definitions as src/03_features.py, so live predictions match what the model was trained on."""
import os, sys
import pandas as pd, numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (team_form_features, h2h_ppg, fill_season_start, add_diff_features, squad_floor,
                      SQUAD, H2H_DEFAULT, DEV_END)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data', 'processed')
SEASONS = ['2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25','2025-26','2026-27']


def season_of(date):
    y = date.year if date.month >= 8 else date.year - 1
    return f'{y}-{str(y + 1)[2:]}'


class Predictor:
    def __init__(self):
        b = self._load_model()
        self.model, self.features, self.classes = b['model'], b['features'], b['classes']
        self.long = pd.read_csv(os.path.join(P, 'team_matches_long.csv'), parse_dates=['Date'])
        self.matches = pd.read_csv(os.path.join(P, 'matches_clean.csv'), parse_dates=['MatchDate'])
        self.squad = pd.read_csv(os.path.join(P, 'squad_strength_by_season.csv'))
        elo = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'matches', 'club_elo_ratings.csv'), parse_dates=['date'])
        self.elo = elo[elo.country == 'ENG']
        # Same gap-filling rule as training (src/03_features.py): median of the development seasons (<= 2023-24).
        mf = pd.read_csv(os.path.join(P, 'model_features.csv'), usecols=['Season'] + list(self.features))
        self.fill = mf.loc[mf.Season <= DEV_END, self.features].median()
        # the deployed model is fitted on every completed season up to 2025-26
        self.n_train = int((mf.Season <= '2025-26').sum())
        self.train_span = f"{mf.Season.min()} → 2025-26"

    @staticmethod
    def _load_model():
        """Load the saved model. If this computer has a different scikit-learn version, the saved file
        may not load cleanly, so retrain the champion (same settings, same data) in a few seconds instead."""
        import warnings, json
        path = os.path.join(ROOT, 'models', 'best_model.pkl')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')          # treat version-mismatch warnings as failure
                return joblib.load(path)
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            best = json.load(open(os.path.join(ROOT, 'results', 'best_params.json')))
            assert best['champion'] == 'Random Forest'
            f = pd.read_csv(os.path.join(P, 'model_features.csv'))
            cols = list(dict.fromkeys(['HomeElo','AwayElo','Elo_diff','H2H_HomePPG','Home_promoted','Away_promoted'] +
                                      [c for c in pd.read_csv(os.path.join(P, 'feature_list.csv')).feature if c.endswith('_diff')]))
            d = f[f.Season <= '2025-26']
            m = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, **best['Random Forest']['params']).fit(d[cols], d.Result)
            return dict(model=m, features=cols, classes=['Home win', 'Draw', 'Away win'])

    # ---------- features: the SAME code as training (src/features.py) ----------
    def _form(self, home, away, date):
        """Form features of both teams just before `date`: their earlier matches plus one still-to-play row each,
        run through the same function that builds the training features."""
        pair = [home, away]
        hist = self.long[self.long.Team.isin(pair) & (self.long.Date < date)]
        fixture = pd.DataFrame({'MatchID': [-1, -1], 'Date': [date, date], 'Season': [season_of(date)] * 2,
                                'Team': pair, 'Opponent': [away, home], 'IsHome': [1, 0]})
        feats = team_form_features(pd.concat([hist, fixture], ignore_index=True))
        out = {}
        for team in pair:
            row = fill_season_start(feats.loc[[(-1, team)]].reset_index(drop=True)).iloc[0].to_dict()
            h = hist[hist.Team == team].sort_values(['Date', 'MatchID'])
            row['last5'] = h.tail(5)[['Date', 'Opponent', 'IsHome', 'GF', 'GA', 'Points']]     # for display only
            out[team] = row
        return out[home], out[away]

    def elo_on(self, team, date):
        e = self.elo[(self.elo.club == team) & (self.elo.date <= date)]
        if team == "Nott'm Forest":  # the Elo file lists Forest under two spellings
            e = pd.concat([e, self.elo[(self.elo.club == 'Nottm Forest') & (self.elo.date <= date)]])
            last = self.matches[(self.matches.MatchDate < date) & ((self.matches.HomeTeam == team) | (self.matches.AwayTeam == team))]
            if len(last):  # keep the spelling whose rating matches the match data
                r = last.iloc[-1]; ref = r.HomeElo if r.HomeTeam == team else r.AwayElo
                e = e[e.date == e.date.max()]
                e = e.iloc[[(e.elo - ref).abs().argmin()]]
        return float(e.sort_values('date').elo.iloc[-1]) if len(e) else np.nan

    def prev_squad(self, team, date):
        s = season_of(date); prev = SEASONS[SEASONS.index(s) - 1]
        sq = self.squad[self.squad.Season == prev].set_index('Team')
        if team in sq.index:
            return sq.loc[team, SQUAD].to_dict(), 0
        return squad_floor(sq).to_dict(), 1  # promoted: weakest-3 average (same rule as src/02_clean_merge.py)

    def h2h(self, home, away, date):
        m = self.matches
        past = m[(m.MatchDate < date) & (((m.HomeTeam == home) & (m.AwayTeam == away)) | ((m.HomeTeam == away) & (m.AwayTeam == home)))].tail(5)
        v = h2h_ppg(list(zip(past.HomeTeam, past.PointsHome)), home)
        return (H2H_DEFAULT if np.isnan(v) else v), past

    # ---------- full feature row ----------
    def features_for(self, home, away, date=None, home_elo=None, away_elo=None):
        date = pd.Timestamp(date) if date is not None else self.matches.MatchDate.max() + pd.Timedelta(days=1)
        H, A = self._form(home, away, date)
        he = home_elo if home_elo is not None else self.elo_on(home, date)
        ae = away_elo if away_elo is not None else self.elo_on(away, date)
        hs, hp = self.prev_squad(home, date); as_, ap = self.prev_squad(away, date)
        h2h, past = self.h2h(home, away, date)
        row = dict(HomeElo=he, AwayElo=ae, H2H_HomePPG=h2h, Home_promoted=hp, Away_promoted=ap, Month=date.month)
        for side, st, sq in (('Home', H, hs), ('Away', A, as_)):
            row.update({f'{side}_{c}': v for c, v in st.items() if c != 'last5'})
            row.update({f'{side}_prev_{c}': sq[c] for c in SQUAD})
        m = add_diff_features(pd.DataFrame([row]))          # same difference features as training
        return m[self.features], dict(home=H, away=A, h2h_matches=past, home_elo=he, away_elo=ae, home_promoted=hp, away_promoted=ap)

    def predict(self, home, away, date=None):
        X, info = self.features_for(home, away, date)
        X = X.fillna(self.fill)
        p = self.model.predict_proba(X)[0]
        return dict(zip(self.classes, p)), info

    def current_teams(self):
        last = self.matches.Season.max()
        m = self.matches[self.matches.Season == last]
        return sorted(set(m.HomeTeam) | set(m.AwayTeam))


if __name__ == '__main__':
    pr = Predictor()
    p, info = pr.predict('Arsenal', 'Man City')
    print({k: round(v, 3) for k, v in p.items()})
