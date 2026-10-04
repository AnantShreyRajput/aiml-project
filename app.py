"""Step 6: Premier League match predictor app.  Run:  streamlit run app.py"""
import os, sys
import pandas as pd
import altair as alt
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from predict import Predictor

st.set_page_config(page_title='EPL Match Predictor', page_icon='⚽', layout='wide')
COLORS = {'Home win': '#2a78d6', 'Draw': '#eb6834', 'Away win': '#1baf7a'}


@st.cache_resource
def load():
    return Predictor()


pr = load()
teams = pr.current_teams()
last_date = pr.matches.MatchDate.max()

st.title('⚽ Premier League Match Predictor')
st.caption(f'Random Forest trained on {len(pr.matches):,} EPL matches (2016-17 → 2026-27). '
           f'Data up to {last_date:%d %b %Y}.')

tab_pred, tab_perf, tab_about = st.tabs(['Predict a match', 'Model performance', 'How it works'])

# ------------------------------------------------------------------ predict
with tab_pred:
    c1, c2 = st.columns(2)
    home = c1.selectbox('🏠 Home team', teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 0)
    away = c2.selectbox('✈️ Away team', [t for t in teams if t != home],
                        index=0)

    probs, info = pr.predict(home, away)
    pick = max(probs, key=probs.get)
    label = {'Home win': f'{home} win', 'Draw': 'Draw', 'Away win': f'{away} win'}

    st.subheader(f'{home} vs {away}')
    m1, m2, m3 = st.columns(3)
    m1.metric(f'{home} win', f"{probs['Home win']:.0%}")
    m2.metric('Draw', f"{probs['Draw']:.0%}")
    m3.metric(f'{away} win', f"{probs['Away win']:.0%}")

    df = pd.DataFrame({'Outcome': [label[k] for k in probs], 'key': list(probs), 'Probability': list(probs.values())})
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=28).encode(
        x=alt.X('Probability:Q', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1]), title=None),
        y=alt.Y('Outcome:N', sort=list(df.Outcome), title=None),
        color=alt.Color('key:N', scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None),
        tooltip=['Outcome', alt.Tooltip('Probability:Q', format='.1%')]).properties(height=150)
    st.altair_chart(chart, width='stretch')

    conf = probs[pick]
    level = 'high' if conf >= 0.6 else 'medium' if conf >= 0.5 else 'low'
    cb = pd.read_csv(os.path.join(ROOT, 'results', 'accuracy_by_confidence.csv'))
    band = '<40%' if conf < .4 else '40-50%' if conf < .5 else '50-60%' if conf < .6 else '60-70%' if conf < .7 else '70%+'
    hit = cb.set_index('band').loc[band, 'accuracy']
    st.info(f"**Most likely: {label[pick]}** ({conf:.0%}) — confidence is **{level}**. "
            f"In testing, picks with {band} confidence were right {hit:.0%} of the time.")

    # ---- why: side-by-side comparison
    st.markdown('#### Why? Key numbers the model looks at')
    H, A = info['home'], info['away']
    sq = pr.squad[pr.squad.Season == '2025-26'].set_index('Team')
    val = lambda t: f"£{sq.loc[t, 'squad_total_cost']:.0f}m" if t in sq.index else 'promoted'
    comp = pd.DataFrame({
        'Stat': ['Elo rating', 'Points per game (last 5)', 'Goals scored per game (last 5)',
                 'Goals conceded per game (last 5)', 'Shots on target per game (last 5)',
                 'Points per game this season', 'Last season FPL squad value'],
        home: [f"{info['home_elo']:.0f}", f"{H['Points_L5']:.2f}", f"{H['GF_L5']:.2f}", f"{H['GA_L5']:.2f}",
               f"{H['SoT_L5']:.1f}", f"{H['SeasonPPG']:.2f}", val(home)],
        away: [f"{info['away_elo']:.0f}", f"{A['Points_L5']:.2f}", f"{A['GF_L5']:.2f}", f"{A['GA_L5']:.2f}",
               f"{A['SoT_L5']:.1f}", f"{A['SeasonPPG']:.2f}", val(away)],
    })
    st.dataframe(comp, hide_index=True, width='stretch')

    def form_table(s):
        t = s['last5'].copy().iloc[::-1]
        t['Result'] = t.Points.map({3: 'W', 1: 'D', 0: 'L'})
        t['Score'] = t.GF.astype(int).astype(str) + '-' + t.GA.astype(int).astype(str)
        t['Venue'] = t.IsHome.map({1: 'H', 0: 'A'})
        t['Date'] = t.Date.dt.strftime('%d %b %Y')
        return t[['Date', 'Opponent', 'Venue', 'Score', 'Result']]

    f1, f2 = st.columns(2)
    f1.markdown(f'**{home} — last 5 matches**'); f1.dataframe(form_table(H), hide_index=True, width='stretch')
    f2.markdown(f'**{away} — last 5 matches**'); f2.dataframe(form_table(A), hide_index=True, width='stretch')

    h2h = info['h2h_matches']
    st.markdown('**Head-to-head (last 5 meetings)**')
    if len(h2h):
        t = h2h[['MatchDate', 'HomeTeam', 'FTHome', 'FTAway', 'AwayTeam']].iloc[::-1].copy()
        t['MatchDate'] = t.MatchDate.dt.strftime('%d %b %Y')
        t['Score'] = t.FTHome.astype(int).astype(str) + '-' + t.FTAway.astype(int).astype(str)
        st.dataframe(t[['MatchDate', 'HomeTeam', 'Score', 'AwayTeam']].rename(columns={'MatchDate': 'Date'}),
                     hide_index=True, width='stretch')
    else:
        st.write('No meetings in the data.')
    if info['home_promoted'] or info['away_promoted']:
        st.caption('A newly promoted team has no last-season EPL squad data, so the model uses the average of the 3 weakest teams.')

# ------------------------------------------------------------------ performance
with tab_perf:
    st.markdown('Tested on **760 matches (2024-25 and 2025-26)** the model never saw during training.')
    res = pd.read_csv(os.path.join(ROOT, 'results', 'model_comparison.csv'))
    res = res[['model', 'test_accuracy', 'test_logloss', 'test_macro_f1']].rename(columns={
        'model': 'Model', 'test_accuracy': 'Accuracy', 'test_logloss': 'Log loss (lower = better)', 'test_macro_f1': 'Macro F1'})
    st.dataframe(res.style.format({'Accuracy': '{:.1%}', 'Log loss (lower = better)': '{:.3f}', 'Macro F1': '{:.3f}'}),
                 hide_index=True, width='stretch')
    figs = os.path.join(ROOT, 'results', 'figures')
    names = sorted(os.listdir(figs)) if os.path.isdir(figs) else []
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for c, n in zip(cols, names[i:i + 2]):
            c.image(os.path.join(figs, n), width='stretch')

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Pipeline:** data collection → cleaning & merging → feature engineering → model training → evaluation → this app.

**Inputs the model uses (27 features, home minus away):**
- Elo ratings and their difference
- Form over the last 3, 5 and 10 matches: points, goals, goal difference, shots, shots on target, wins
- Points and goal difference per game this season, rest days
- Last season's squad strength from Fantasy Premier League data (player prices, points, ICT index by position)
- Head-to-head record over the last 5 meetings, promoted-team flags

**Model:** Random Forest (300 trees, max depth 4), chosen by season-by-season cross-validation.

**Limits:** it never predicts a draw as the single most likely result, and it knows nothing about injuries
on match day, manager changes or team news. Football is random: even bookmakers get only ~52% right.
""")
