# Raw data – EPL match predictor (Step 1)

## matches/
- `epl_matches_2016-17_to_2026-27.csv` – 3,820 EPL matches (10 full seasons + first 20 games of 2026-27): date, teams, full/half-time score, shots, shots on target, fouls, corners, yellow/red cards, bookmaker odds, pre-match Elo, `Season`. Source: football-data.co.uk via GitHub mirror `xgabora/Club-Football-Match-Data-2000-2025`.
- `club_elo_ratings.csv` – Elo rating history of English clubs (used by the app for each team's latest rating).

## players/
- `players_raw_<season>.csv` – one row per player per season from the official Fantasy Premier League API (via `vaastav/Fantasy-Premier-League`): position, team, price, minutes, goals, assists, form, ICT index, injury status. Used to build squad strength.

## squads/
- `master_team_list.csv` – FPL team id → team name for 2016-17 to 2023-24.
- `teams_<season>.csv` – the same mapping for 2024-25 onward.

## Gaps
- No possession % or team-level xG for all seasons, and no FIFA/EA FC ratings (those sources were not reachable). FPL price, points and ICT index stand in as player ratings.
