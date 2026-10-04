"""Build the model's input features for ANY fixture, using only matches played before a given date.
Same definitions as src/03_features.py, so live predictions match what the model was trained on."""
import pandas as pd, numpy as np, joblib, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data', 'processed')
SEASONS = ['2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25','2025-26','2026-27']
SQUAD = ['xi_avg_cost','xi_avg_points','xi_avg_ict','squad_total_cost','team_goals','gk_pts','def_pts','mid_pts','att_pts']


def season_of(date):
    y = date.year if date.month >= 8 else date.year - 1
    return f'{y}-{str(y + 1)[2:]}'


class Predictor:
    def __init__(self):
        b = self._load_model()
        self.model, self.features, self.classes = b['model'], b['features'], b['classes']
        self.long = pd.read_csv(os.path.join(P, 'team_matches_long.csv'), parse_dates=['Date'])
        self.long['Win'] = (self.long.Points == 3).astype(int)
        self.long['GD'] = self.long.GF - self.long.GA
        self.matches = pd.read_csv(os.path.join(P, 'matches_clean.csv'), parse_dates=['MatchDate'])
        self.squad = pd.read_csv(os.path.join(P, 'squad_strength_by_season.csv'))
        elo = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'matches', 'club_elo_ratings.csv'), parse_dates=['date'])
        self.elo = elo[elo.country == 'ENG']

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

    # ---------- team state just before `date` ----------
    def team_state(self, team, date):
        h = self.long[(self.long.Team == team) & (self.long.Date < date)].sort_values(['Date', 'MatchID'])
        s = {}
        for N, cols in ((3, ['Points']), (10, ['Points']), (5, ['Points', 'GF', 'GA', 'GD', 'Shots', 'SoT', 'Win'])):
            for c in cols:
                s[f'{c}_L{N}'] = h[c].tail(N).mean() if len(h) else np.nan
        cur = h[h.Season == season_of(date)]
        s['SeasonPPG'] = cur.Points.mean() if len(cur) else 0.0
        s['SeasonGDpg'] = cur.GD.mean() if len(cur) else 0.0
        s['RestDays'] = min((date - h.Date.iloc[-1]).days, 14) if len(h) else 7
        s['last5'] = h.tail(5)[['Date', 'Opponent', 'IsHome', 'GF', 'GA', 'Points']]
        return s

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
        return sq[SQUAD].apply(lambda c: c.nsmallest(3).mean()).to_dict(), 1  # promoted: weakest-3 average

    def h2h(self, home, away, date):
        m = self.matches
        past = m[(m.MatchDate < date) & (((m.HomeTeam == home) & (m.AwayTeam == away)) | ((m.HomeTeam == away) & (m.AwayTeam == home)))].tail(5)
        if not len(past):
            return 1.35, past
        pts = np.where(past.HomeTeam == home, past.PointsHome, past.PointsAway)
        return float(pts.mean()), past

    # ---------- full feature row ----------
    def features_for(self, home, away, date=None, home_elo=None, away_elo=None):
        date = pd.Timestamp(date) if date is not None else self.matches.MatchDate.max() + pd.Timedelta(days=1)
        H, A = self.team_state(home, date), self.team_state(away, date)
        he = home_elo if home_elo is not None else self.elo_on(home, date)
        ae = away_elo if away_elo is not None else self.elo_on(away, date)
        hs, hp = self.prev_squad(home, date); as_, ap = self.prev_squad(away, date)
        h2h, past = self.h2h(home, away, date)
        f = dict(HomeElo=he, AwayElo=ae, Elo_diff=he - ae, H2H_HomePPG=h2h, Home_promoted=hp, Away_promoted=ap)
        for c in ['Points_L5', 'GF_L5', 'GA_L5', 'GD_L5', 'Shots_L5', 'SoT_L5', 'Win_L5', 'Points_L10', 'SeasonPPG', 'SeasonGDpg', 'RestDays', 'Points_L3']:
            f[f'{c}_diff'] = H[c] - A[c]
        for c in SQUAD:
            f[f'prev_{c}_diff'] = hs[c] - as_[c]
        X = pd.DataFrame([f])[self.features]
        return X, dict(home=H, away=A, h2h_matches=past, home_elo=he, away_elo=ae, home_promoted=hp, away_promoted=ap)

    def predict(self, home, away, date=None):
        X, info = self.features_for(home, away, date)
        X = X.fillna(0)
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
