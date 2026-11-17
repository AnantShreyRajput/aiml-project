"""Step 7: probability-quality and draw evaluation of every model from step 6.
Run from project root:  python src/07_probability_eval.py
 
Reads  results/oof_predictions.csv
Writes results/ablation_per_season.csv, ablation_summary.csv, paired_differences.csv, calibration_by_class.csv,
       results/probability_report.txt, results/figures/7_logloss_by_season.png, 8_calibration_models.png
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
 
R = 'results'
oof = pd.read_csv(f'{R}/oof_predictions.csv')
PC = ['pH', 'pD', 'pA']
CV_SEASONS = ['2021-22', '2022-23', '2023-24']        # used ONLY to choose the draw threshold
TEST_SEASONS = ['2024-25', '2025-26']                  # untouched for every decision
ALL_SEASONS = CV_SEASONS + TEST_SEASONS
EPS = 1e-12
 
 
# ------------------------------------------------------------------ metrics
def logloss(y, p):   return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))
def brier(y, p):     return float(np.mean(((p - np.eye(3)[y]) ** 2).sum(axis=1)))     # multiclass Brier, 0 = perfect, 0.667 = uniform guess
def accuracy(y, p):  return float((p.argmax(1) == y).mean())
 
def ece(y, p, bins=8):
    """Expected calibration error, averaged over the three outcomes (one-vs-rest, equal-count bins)."""
    tot = []
    for k in range(3):
        q = pd.qcut(p[:, k], bins, duplicates='drop')
        d = pd.DataFrame({'p': p[:, k], 'hit': (y == k).astype(float), 'b': q})
        g = d.groupby('b', observed=True).agg(p=('p', 'mean'), hit=('hit', 'mean'), n=('p', 'size'))
        tot.append((g.n / g.n.sum() * (g.p - g.hit).abs()).sum())
    return float(np.mean(tot))
 
def draw_pred(p, delta):
    """Predict a draw when home and away look evenly matched (|pH - pA| < delta); otherwise the likelier side."""
    pred = np.where(p[:, 0] >= p[:, 2], 0, 2)
    return np.where(np.abs(p[:, 0] - p[:, 2]) < delta, 1, pred)
 
def draw_pr(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y != 1)).sum(); fn = ((pred != 1) & (y == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(prec), float(rec), float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0
 
def macro_f1(y, pred):
    f = []
    for k in range(3):
        tp = ((pred == k) & (y == k)).sum(); fp = ((pred == k) & (y != k)).sum(); fn = ((pred != k) & (y == k)).sum()
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))
 
 
def get(df, model, seasons):
    d = df[(df.model == model) & df.Season.isin(seasons)].sort_values('MatchID')
    return d.Result.values.astype(int), d[PC].values, d.MatchID.values
 
 
models = list(dict.fromkeys(oof.model))
 
# ------------------------------------------------------------------ 1. per-season table (uncertainty across seasons)
rows = []
for m in models:
    for s in ALL_SEASONS:
        y, p, _ = get(oof, m, [s])
        rows.append(dict(model=m, season=s, n=len(y), log_loss=logloss(y, p), brier=brier(y, p), accuracy=accuracy(y, p)))
per = pd.DataFrame(rows); per.to_csv(f'{R}/ablation_per_season.csv', index=False)
 
# ------------------------------------------------------------------ 2. summary table
_y_test = get(oof, models[0], TEST_SEASONS)[0]
DRAW_BASE_RATE = float((_y_test == 1).mean())   # share of test matches that are draws = precision of guessing 'draw' every time
summ = []
for m in models:
    ps = per[per.model == m]
    y, p, _ = get(oof, m, TEST_SEASONS)
    # draw threshold chosen on the CV seasons only, then applied unchanged to the test seasons
    yc, pc, _ = get(oof, m, CV_SEASONS)
    grid = np.arange(0.0, 0.40, 0.01)
    delta = grid[int(np.argmax([macro_f1(yc, draw_pred(pc, d)) for d in grid]))]
    pr_arg = draw_pr(y, p.argmax(1)); pr_rule = draw_pr(y, draw_pred(p, delta))
    summ.append(dict(model=m,
        mean_logloss_5seasons=ps.log_loss.mean(), std_logloss_5seasons=ps.log_loss.std(ddof=1),
        test_logloss=logloss(y, p), test_brier=brier(y, p), test_accuracy=accuracy(y, p), test_ece=ece(y, p),
        draw_precision_argmax=pr_arg[0], draw_recall_argmax=pr_arg[1],
        draw_delta=delta, draw_precision_rule=pr_rule[0], draw_recall_rule=pr_rule[1], draw_f1_rule=pr_rule[2],
        test_macro_f1_rule=macro_f1(y, draw_pred(p, delta)),
        test_accuracy_rule=float((draw_pred(p, delta) == y).mean())))
summ = pd.DataFrame(summ).sort_values('test_logloss'); summ.to_csv(f'{R}/ablation_summary.csv', index=False)
 
# ------------------------------------------------------------------ 2b. accuracy by confidence (used by the app)
# Deployed configuration = Random Forest, compact 27 features. Walk-forward predictions (every season predicted by a model
# trained only on earlier seasons) over 5 seasons give 1,900 matches, so the bands are less noisy than the 760 test matches
# and do not depend on the refitted production model, which has seen the test seasons.
y_rf, p_rf, _ = get(oof, 'RF | C Elo+Form+FPL', ALL_SEASONS)
conf_rf = p_rf.max(1)
band = pd.cut(conf_rf, [0, 0.4, 0.5, 0.6, 0.7, 1], labels=['<40%', '40-50%', '50-60%', '60-70%', '70%+'])
cb = pd.DataFrame({'band': band, 'hit': (p_rf.argmax(1) == y_rf).astype(float), 'conf': conf_rf}).groupby('band', observed=True) \
       .agg(accuracy=('hit', 'mean'), matches=('hit', 'size'), avg_conf=('conf', 'mean'))
cb.to_csv(f'{R}/accuracy_by_confidence_walkforward.csv')

# ------------------------------------------------------------------ 3. paired bootstrap of log-loss differences
rng = np.random.default_rng(42)
def per_match_loss(m, seasons):
    y, p, ids = get(oof, m, seasons); return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)), ids
 
PAIRS = [('LR | B Elo+Form', 'LR | A Elo only'), ('LR | C Elo+Form+FPL', 'LR | B Elo+Form'), ('LR | C Elo+Form+FPL', 'LR | A Elo only'),
         ('LR | D Elo+FPL', 'LR | A Elo only'), ('RF | B Elo+Form', 'RF | A Elo only'), ('RF | C Elo+Form+FPL', 'RF | B Elo+Form'),
         ('RF | C Elo+Form+FPL', 'RF | A Elo only'), ('Dixon-Coles (Poisson)', 'LR | A Elo only'),
         ('LR | C Elo+Form+FPL', 'Dixon-Coles (Poisson)'), ('Bookmakers (reference)', 'LR | C Elo+Form+FPL')]
pd_rows = []
for label, seasons in (('test 2024-26 (760)', TEST_SEASONS), ('all 5 seasons (1900)', ALL_SEASONS)):
    for a, b in PAIRS:
        la, ia = per_match_loss(a, seasons); lb, ib = per_match_loss(b, seasons)
        assert (ia == ib).all()
        diff = la - lb                                           # negative = model `a` is better than `b`
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
        per_a = per[(per.model == a)].set_index('season').log_loss; per_b = per[(per.model == b)].set_index('season').log_loss
        sel = [s for s in ALL_SEASONS if s in seasons]
        wins = int((per_a[sel] < per_b[sel]).sum())
        pd_rows.append(dict(evaluated_on=label, model_a=a, model_b=b, mean_diff_logloss=diff.mean(),
                            ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5),
                            seasons_a_better=f'{wins}/{len(seasons)}',
                            significant=bool(np.percentile(bs, 97.5) < 0 or np.percentile(bs, 2.5) > 0)))
pdf = pd.DataFrame(pd_rows); pdf.to_csv(f'{R}/paired_differences.csv', index=False)
 
# ------------------------------------------------------------------ 4. calibration tables + figure
KEY = ['LR | A Elo only', 'LR | C Elo+Form+FPL', 'RF | C Elo+Form+FPL', 'Dixon-Coles (Poisson)']
COL = {'LR | A Elo only': '#b9b8b1', 'LR | C Elo+Form+FPL': '#2a78d6', 'RF | C Elo+Form+FPL': '#1baf7a', 'Dixon-Coles (Poisson)': '#eb6834'}
cal_rows = []
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for k, name in enumerate(['Home win', 'Draw', 'Away win']):
    ax = axes[k]; ax.plot([0, 1], [0, 1], '--', color='#999', lw=1)
    for m in KEY:
        y, p, _ = get(oof, m, TEST_SEASONS)
        q = pd.qcut(p[:, k], 6, duplicates='drop')
        g = pd.DataFrame({'p': p[:, k], 'hit': (y == k).astype(float), 'b': q}).groupby('b', observed=True).agg(p=('p', 'mean'), hit=('hit', 'mean'), n=('p', 'size'))
        ax.plot(g.p, g.hit, marker='o', color=COL[m], label=m, lw=1.8)
        cal_rows.append(g.reset_index(drop=True).assign(model=m, outcome=name))
    ax.set_title(f'{name}'); ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency'); ax.grid(color='#e6e5e0')
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle('Calibration on test seasons 2024-25 and 2025-26 (closer to the dashed line = better)')
plt.tight_layout(); plt.savefig(f'{R}/figures/8_calibration_models.png', dpi=150); plt.close()
pd.concat(cal_rows).to_csv(f'{R}/calibration_by_class.csv', index=False)
 
fig, ax = plt.subplots(figsize=(8, 4.5))
for m in ['Baseline: always home win', 'LR | A Elo only', 'LR | B Elo+Form', 'LR | C Elo+Form+FPL', 'Dixon-Coles (Poisson)', 'Bookmakers (reference)']:
    d = per[per.model == m]; ax.plot(d.season, d.log_loss, marker='o', label=m, lw=1.8)
ax.set_ylabel('Log loss (lower = better)'); ax.set_title('Log loss by season (each model trained only on earlier seasons)')
ax.grid(color='#e6e5e0'); ax.legend(fontsize=8, frameon=False); plt.xticks(rotation=30); plt.tight_layout()
plt.savefig(f'{R}/figures/7_logloss_by_season.png', dpi=150); plt.close()
 
# ------------------------------------------------------------------ 5. readable report
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
rep = ['PROBABILITY AND DRAW EVALUATION',
       'Walk-forward: every season is predicted by models trained only on earlier seasons.',
       'Test = 2024-25 + 2025-26 (760 matches). Draw threshold (delta) was chosen on 2021-22..2023-24 only.',
       f'Draw base rate on the test seasons = {DRAW_BASE_RATE:.3f}: a draw precision near this value means the draw rule has no skill beyond guessing.',
       'test_accuracy_rule = accuracy when the draw rule is applied (the cost of predicting draws).', '',
       '--- Summary (sorted by test log loss) ---',
       summ.round(4).to_string(index=False), '',
       '--- Per-season log loss ---',
       per.pivot(index='model', columns='season', values='log_loss').round(4).to_string(), '',
       '--- Paired differences in log loss (negative = model_a better), 95% bootstrap CI ---',
       pdf.round(4).to_string(index=False)]
open(f'{R}/probability_report.txt', 'w').write('\n'.join(rep)); print('\n'.join(rep))
