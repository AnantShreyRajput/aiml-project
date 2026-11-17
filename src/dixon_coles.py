"""Dixon-Coles (1997) goal model: a football-specific baseline.
 
Each team gets an attack and a defence strength; goals are Poisson distributed:
    home goals ~ Poisson(exp(home_adv + attack_home - defence_away))
    away goals ~ Poisson(exp(attack_away - defence_home))
A small correction (rho) fixes the under-estimated 0-0, 1-0, 0-1 and 1-1 scorelines, and older matches are
down-weighted exponentially (time decay).  Match probabilities H/D/A are summed from the scoreline grid.
Only matches played BEFORE the fit date are used, so there is no leakage.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
 
MAX_GOALS = 10
 
 
def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction factor (vectorised)."""
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t
 
 
class DixonColes:
    def __init__(self, xi=0.0019, l2=0.01):
        self.xi = xi      # time decay per day (0.0019 -> half-life of about one year)
        self.l2 = l2      # tiny ridge penalty keeps ratings of rarely seen teams near the average
 
    def fit(self, df, as_of):
        """df needs columns MatchDate, HomeTeam, AwayTeam, FTHome, FTAway. Uses only rows dated before `as_of`."""
        d = df[df.MatchDate < as_of]
        self.teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = d.HomeTeam.map(idx).values
        ai = d.AwayTeam.map(idx).values
        x = d.FTHome.values.astype(int)
        y = d.FTAway.values.astype(int)
        w = np.exp(-self.xi * (as_of - d.MatchDate).dt.days.values)
 
        def nll(p):
            att, dfn, h, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            lam = np.exp(h + att[hi] - dfn[ai])
            mu = np.exp(att[ai] - dfn[hi])
            tau = np.clip(_tau(x, y, lam, mu, rho), 1e-6, None)
            ll = w * (np.log(tau) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu))
            pen = self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            return -ll.sum() + pen
 
        p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.3, 0.3)]
        r = minimize(nll, p0, method='L-BFGS-B', bounds=bounds)
        self.att = dict(zip(self.teams, r.x[:n]))
        self.dfn = dict(zip(self.teams, r.x[n:2 * n]))
        self.h, self.rho = r.x[2 * n], r.x[2 * n + 1]
        return self
 
    def predict_proba(self, home, away):
        """[P(home win), P(draw), P(away win)]. A team never seen before gets average strength (0)."""
        lam = np.exp(self.h + self.att.get(home, 0.0) - self.dfn.get(away, 0.0))
        mu = np.exp(self.att.get(away, 0.0) - self.dfn.get(home, 0.0))
        g = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)):
            grid[i, j] *= float(_tau(np.array(i), np.array(j), np.array(lam), np.array(mu), self.rho))
        grid /= grid.sum()
        return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
 
 
def walk_forward(matches, target_idx, refit='month'):
    """Predict each match in `matches.loc[target_idx]` with a model fitted only on earlier matches.
    The model is refitted at the start of every calendar month (cheap, and still strictly past-only)."""
    out = {}
    tgt = matches.loc[target_idx].copy()
    tgt['_ym'] = tgt.MatchDate.dt.to_period('M')
    for _, g in tgt.groupby('_ym'):
        as_of = g.MatchDate.min().normalize()           # everything strictly before the first match of the month
        model = DixonColes().fit(matches, as_of)
        for i, r in g.iterrows():
            out[i] = model.predict_proba(r.HomeTeam, r.AwayTeam)
    return pd.DataFrame.from_dict(out, orient='index', columns=['pH', 'pD', 'pA']).loc[target_idx]
