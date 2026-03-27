# PokerStars Hand History Report

A simple Python script that parses PokerStars hand histories and generates per-hand results, summary stats, all-in EV output, and a matplotlib equity curve.

## Features

- Parses raw PokerStars hand history text
- Exports one row per hand to CSV
- Calculates core preflop stats:
  - VPIP
  - PFR
  - 3-bet rate
- Reports bb/100 overall and by position
- Tracks simple leak counters:
  - open limps
  - flat calls vs raises
  - BB defend calls
- Exports eligible all-in EV spots
- Generates an equity curve chart with matplotlib

## Output

Running the script creates:

- `hands_report.csv` — hand-by-hand results and flags
- `allin_ev.csv` — eligible all-in EV spots
- `equity_curve.png` — cumulative winnings in big blinds

## Requirements

- Python 3.9+
- `pandas`
- `matplotlib`

Install dependencies:

```bash
pip install pandas matplotlib
