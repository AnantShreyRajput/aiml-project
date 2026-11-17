"""Step 1: check the raw data. Run from project root:  python src/01_check_raw_data.py

The raw data is downloaded by hand (public sources, no login). This script makes Step 1 reproducible:
it verifies that every raw file is present and byte-for-byte the one the reported results were produced from
(SHA-256 + row count recorded in data/raw/MANIFEST.csv), and it prints where each file comes from if something is missing.

    python src/01_check_raw_data.py                    # verify (exit code 1 if anything is missing or different)
    python src/01_check_raw_data.py --write-manifest   # record the current raw files as the reference (after a deliberate data update)
"""
import hashlib, os, sys
import pandas as pd

RAW = 'data/raw'
MANIFEST = f'{RAW}/MANIFEST.csv'
SEASONS = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27']

SOURCES = {
    'matches/epl_matches_2016-17_to_2026-27.csv': 'EPL results, shots, cards, odds, pre-match Elo: football-data.co.uk (England, Premier League), '
                                                  'via the GitHub mirror xgabora/Club-Football-Match-Data-2000-2025, filtered to the Premier League.',
    'matches/club_elo_ratings.csv': 'Elo rating history of English clubs (club / country / elo / date columns, clubelo.com format), as downloaded for this project.',
    'players/players_raw_<season>.csv': 'Official Fantasy Premier League player data, file players_raw.csv of each season in '
                                        'github.com/vaastav/Fantasy-Premier-League (data/<season>/players_raw.csv).',
    'squads/master_team_list.csv': 'FPL team id -> team name, 2016-17 to 2023-24: vaastav/Fantasy-Premier-League (data/master_team_list.csv).',
    'squads/teams_<season>.csv': 'FPL team id -> team name for 2024-25 onwards: vaastav/Fantasy-Premier-League (data/<season>/teams.csv).',
}


def expected_files():
    files = ['matches/epl_matches_2016-17_to_2026-27.csv', 'matches/club_elo_ratings.csv', 'squads/master_team_list.csv']
    files += [f'players/players_raw_{s}.csv' for s in SEASONS]
    files += [f'squads/teams_{s}.csv' for s in SEASONS[-3:]]
    return files


def describe(rel):
    p = f'{RAW}/{rel}'
    data = open(p, 'rb').read()
    return dict(file=rel, sha256=hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest(),   # line endings do not matter
                rows=len(pd.read_csv(p)))


def main():
    files = expected_files()
    if '--write-manifest' in sys.argv:
        missing = [f for f in files if not os.path.exists(f'{RAW}/{f}')]
        if missing:
            sys.exit(f'cannot write manifest, missing: {missing}')
        pd.DataFrame([describe(f) for f in files]).to_csv(MANIFEST, index=False)
        print(f'wrote {MANIFEST} ({len(files)} files)')
        return
    ref = pd.read_csv(MANIFEST).set_index('file') if os.path.exists(MANIFEST) else None
    problems = []
    for rel in files:
        if not os.path.exists(f'{RAW}/{rel}'):
            problems.append(f'MISSING   {rel}')
            continue
        d = describe(rel)
        if ref is not None and rel in ref.index and (d['sha256'] != ref.loc[rel, 'sha256'] or d['rows'] != ref.loc[rel, 'rows']):
            problems.append(f'CHANGED   {rel}  (rows {d["rows"]} vs {ref.loc[rel, "rows"]} in the manifest)')
        else:
            print(f'[ok] {rel:48s} {d["rows"]:6d} rows')
    m = pd.read_csv(f'{RAW}/matches/epl_matches_2016-17_to_2026-27.csv')
    full = m[m.Season != '2026-27'].groupby('Season').size()
    if not (full == 380).all() or len(full) != 10:
        problems.append(f'match file: expected 10 complete seasons of 380 matches, got {full.to_dict()}')
    print(f'match file: {len(m)} rows = {len(full)} complete seasons x 380 + {int((m.Season == "2026-27").sum())} games of 2026-27')
    if problems:
        print('\nPROBLEMS:'); [print('  ' + p) for p in problems]
        print('\nWhere the files come from:'); [print(f'  {k}: {v}') for k, v in SOURCES.items()]
        sys.exit(1)
    print('\nALL RAW FILES PRESENT AND UNCHANGED')


if __name__ == '__main__':
    main()
