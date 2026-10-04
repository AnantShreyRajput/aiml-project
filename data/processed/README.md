# Processed data (Step 2) – produced by `src/02_clean_merge.py`

| File | Rows | What it is |
|---|---|---|
| `matches_clean.csv` | 3,820 | Cleaned matches, 2016-17 → 2026-27. Team names unified, junk columns dropped, target `Result` (0=Home win, 1=Draw, 2=Away win), implied bookmaker probabilities `ImpHome/Draw/Away`. |
| `team_matches_long.csv` | 7,640 | One row per team per match (goals for/against, shots, corners, cards, points, Elo). Step 3 builds rolling "last 5 matches" form from this. |
| `squad_strength_by_season.csv` | 220 | Per team per season: average price/points/ICT of the 11 most-used players, squad value, goals/assists, GK/DEF/MID/ATT points, current injured-player count. |
| `model_dataset_v1.csv` | 3,440 | Match table + **previous-season** squad strength for home and away team. 2017-18 onward (2016-17 has no earlier season). No missing values. |

## Decisions
- **No leakage:** squad strength comes from the *previous* season, because a season's end-of-year totals contain information about that season's own matches.
- **Promoted teams** (3 per season) have no prior EPL record: flagged with `Home_promoted` / `Away_promoted` and filled with the average of the 3 weakest teams that season.
- `team_xg` dropped: only exists from 2022-23.
- Team-name fixes: Man Utd → Man United, Spurs → Tottenham, Nottm Forest → Nott'm Forest.
- 2019-20/2020-21 COVID seasons are assigned correctly (each has 380 matches).
- Home win 44.2% / draw 23.3% / away win 32.5% — baseline to beat.
- Bookmaker odds are kept for comparison; **don't use them as model input** if you want to claim your model learned football on its own.

---
# Step 3 – features (`src/03_features.py`)
- `model_features.csv` – 3,440 matches x 111 features + `Result` target + bookmaker odds (odds are NOT features, only for comparison). 0 missing values.
- `feature_list.csv` – names of the 111 model inputs.

Feature groups (all use only matches played BEFORE the match date):
- **Rolling form** (last 3/5/10 matches): points, goals for/against, goal difference, shots, shots on target, corners, fouls, cards, wins, clean sheets.
- **Venue form**: last 5 home games for the home team, last 5 away games for the away team.
- **Streaks**: win streak, unbeaten run, loss streak. **Rest days** since last match.
- **Season-to-date**: points per game, goal difference per game, games played.
- **Elo** ratings and `Elo_diff`. **Head-to-head**: home team's points per game in last 5 meetings.
- **Squad strength** from previous season (cost, points, ICT, GK/DEF/MID/ATT).
- **Difference features** (home minus away) and attack-vs-defence matchups.
Checked by hand against Arsenal's real history (form, venue form, streaks, season PPG, rest days all match).
