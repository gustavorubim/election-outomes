"""Generate a synthetic-but-plausible Senate state panel for cycles 2014-2024.

Honesty note: results here are deterministic procedural draws calibrated to roughly
match aggregate historical patterns (D/R seat swings by cycle, state partisan lean).
They are NOT actual historical Senate election results. The panel exists to exercise
the rolling-origin harness at scale; production runs should ingest real returns from
MIT Election Lab or OpenSecrets.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "senate_state_panel.csv"

CYCLES: list[int] = [2014, 2016, 2018, 2020, 2022, 2024]
FORECAST_ONLY_CYCLES: set[int] = set()

# Senate Class assignments. Class I up in 2024/2018/2012; Class II up in 2020/2014/2008;
# Class III up in 2022/2016/2010. Source: Senate.gov.
CLASS_I: list[str] = [
    "AZ",
    "CA",
    "CT",
    "DE",
    "FL",
    "HI",
    "IN",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NE",
    "NV",
    "NJ",
    "NM",
    "NY",
    "ND",
    "OH",
    "PA",
    "RI",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
CLASS_II: list[str] = [
    "AL",
    "AK",
    "AR",
    "CO",
    "DE",
    "GA",
    "IA",
    "ID",
    "IL",
    "KS",
    "KY",
    "LA",
    "MA",
    "ME",
    "MI",
    "MN",
    "MS",
    "MT",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NC",
    "OK",
    "OR",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "VA",
    "WV",
    "WY",
]
CLASS_III: list[str] = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "MD",
    "MO",
    "NV",
    "NH",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "SC",
    "SD",
    "UT",
    "VT",
    "WA",
    "WI",
]

CYCLE_TO_CLASS: dict[int, list[str]] = {
    2014: CLASS_II,
    2016: CLASS_III,
    2018: CLASS_I,
    2020: CLASS_II,
    2022: CLASS_III,
    2024: CLASS_I,
}

CYCLE_CLASS_LABEL: dict[int, str] = {
    2014: "II",
    2016: "III",
    2018: "I",
    2020: "II",
    2022: "III",
    2024: "I",
}

ELECTION_DATE: dict[int, str] = {
    2014: "2014-11-04",
    2016: "2016-11-08",
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
    2024: "2024-11-05",
}

# Approximate state partisan lean in pp (D positive). Calibrated from public sources
# such as Cook PVI; values are illustrative and synthetic for fixture purposes.
STATE_LEAN: dict[str, float] = {
    "AL": -15.0,
    "AK": -9.0,
    "AZ": -2.0,
    "AR": -15.0,
    "CA": +13.0,
    "CO": +4.0,
    "CT": +11.0,
    "DE": +12.0,
    "FL": -3.0,
    "GA": -1.0,
    "HI": +18.0,
    "ID": -19.0,
    "IL": +9.0,
    "IN": -9.0,
    "IA": -5.0,
    "KS": -10.0,
    "KY": -16.0,
    "LA": -12.0,
    "ME": +5.0,
    "MD": +20.0,
    "MA": +20.0,
    "MI": +1.0,
    "MN": +2.0,
    "MS": -10.0,
    "MO": -10.0,
    "MT": -11.0,
    "NE": -13.0,
    "NV": +1.0,
    "NH": +1.0,
    "NJ": +9.0,
    "NM": +5.0,
    "NY": +12.0,
    "NC": -3.0,
    "ND": -17.0,
    "OH": -6.0,
    "OK": -20.0,
    "OR": +9.0,
    "PA": -1.0,
    "RI": +14.0,
    "SC": -8.0,
    "SD": -12.0,
    "TN": -13.0,
    "TX": -5.0,
    "UT": -13.0,
    "VT": +20.0,
    "VA": +3.0,
    "WA": +9.0,
    "WV": -25.0,
    "WI": +0.0,
    "WY": -25.0,
}

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

# Approximate D environment per cycle (pp swing toward D). Sourced informally from
# popular-vote senate national share trends: 2014/2016 R-leaning; 2018 D wave;
# 2020 split; 2022 D held; 2024 R wave.
CYCLE_D_ENVIRONMENT: dict[int, float] = {
    2014: -3.0,
    2016: -1.0,
    2018: +5.0,
    2020: +1.0,
    2022: +1.5,
    2024: -2.5,
}

# Approximate D economic index per cycle (synthetic).
CYCLE_ECONOMY: dict[int, float] = {
    2014: -0.2,
    2016: -0.1,
    2018: +0.1,
    2020: -0.3,
    2022: -0.4,
    2024: -0.2,
}

# Pseudo-incumbents: states where the 2010-baseline incumbent party is locked in.
# For cycles before 2024, we re-derive from prior result alternations.
SEED_INCUMBENT_PARTY: dict[str, str] = {
    "AL": "REP",
    "AK": "REP",
    "AZ": "REP",
    "AR": "REP",
    "CA": "DEM",
    "CO": "DEM",
    "CT": "DEM",
    "DE": "DEM",
    "FL": "REP",
    "GA": "REP",
    "HI": "DEM",
    "ID": "REP",
    "IL": "DEM",
    "IN": "REP",
    "IA": "REP",
    "KS": "REP",
    "KY": "REP",
    "LA": "REP",
    "ME": "REP",
    "MD": "DEM",
    "MA": "DEM",
    "MI": "DEM",
    "MN": "DEM",
    "MS": "REP",
    "MO": "REP",
    "MT": "REP",
    "NE": "REP",
    "NV": "DEM",
    "NH": "DEM",
    "NJ": "DEM",
    "NM": "DEM",
    "NY": "DEM",
    "NC": "REP",
    "ND": "REP",
    "OH": "REP",
    "OK": "REP",
    "OR": "DEM",
    "PA": "REP",
    "RI": "DEM",
    "SC": "REP",
    "SD": "REP",
    "TN": "REP",
    "TX": "REP",
    "UT": "REP",
    "VT": "DEM",
    "VA": "DEM",
    "WA": "DEM",
    "WV": "DEM",
    "WI": "REP",
    "WY": "REP",
}

# Approximate state population scale → registered voters.
STATE_REGISTERED_VOTERS: dict[str, int] = {
    "AL": 3_700_000,
    "AK": 580_000,
    "AZ": 4_500_000,
    "AR": 1_900_000,
    "CA": 22_000_000,
    "CO": 4_000_000,
    "CT": 2_400_000,
    "DE": 800_000,
    "FL": 14_000_000,
    "GA": 7_500_000,
    "HI": 850_000,
    "ID": 1_100_000,
    "IL": 8_500_000,
    "IN": 4_900_000,
    "IA": 2_300_000,
    "KS": 1_900_000,
    "KY": 3_500_000,
    "LA": 3_000_000,
    "ME": 1_100_000,
    "MD": 4_300_000,
    "MA": 4_900_000,
    "MI": 7_700_000,
    "MN": 3_900_000,
    "MS": 2_000_000,
    "MO": 4_400_000,
    "MT": 800_000,
    "NE": 1_300_000,
    "NV": 2_100_000,
    "NH": 1_000_000,
    "NJ": 6_500_000,
    "NM": 1_400_000,
    "NY": 13_000_000,
    "NC": 7_500_000,
    "ND": 530_000,
    "OH": 8_100_000,
    "OK": 2_300_000,
    "OR": 3_000_000,
    "PA": 9_000_000,
    "RI": 800_000,
    "SC": 3_700_000,
    "SD": 600_000,
    "TN": 4_500_000,
    "TX": 17_000_000,
    "UT": 1_900_000,
    "VT": 500_000,
    "VA": 6_000_000,
    "WA": 5_000_000,
    "WV": 1_300_000,
    "WI": 4_400_000,
    "WY": 290_000,
}


def _seeded_random(seed: str) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng


def _incumbency_for_cycle(cycle: int, state: str) -> str:
    """Decide incumbent party deterministically with occasional flips on big swings."""
    base = SEED_INCUMBENT_PARTY[state]
    rng = _seeded_random(f"sen-incumb-{state}-{cycle}")
    flip_probability = max(0.0, 0.18 - abs(STATE_LEAN[state]) / 80.0)
    return ("REP" if base == "DEM" else "DEM") if rng.random() < flip_probability else base


def _generate_row(cycle: int, state: str) -> dict[str, object]:
    rng = _seeded_random(f"sen-{state}-{cycle}")
    incumbent_party = _incumbency_for_cycle(cycle, state)
    incumbent_advantage = 3.5 if incumbent_party in {"DEM", "REP"} else 0.0

    state_lean_pp = STATE_LEAN[state]
    environment_pp = CYCLE_D_ENVIRONMENT[cycle]
    candidate_quality_pp = (rng.random() - 0.5) * 5.0  # ±2.5pp

    incumbent_pp = incumbent_advantage if incumbent_party == "DEM" else -incumbent_advantage
    expected_d_share_pp = (
        50.0 + state_lean_pp + environment_pp + candidate_quality_pp + incumbent_pp
    )
    actual_d_pp = expected_d_share_pp + (rng.random() - 0.5) * 4.0
    actual_d = max(0.30, min(0.72, actual_d_pp / 100.0))
    actual_r = round(1.0 - actual_d, 4)
    actual_d = round(actual_d, 4)

    poll_error_pp = (rng.random() - 0.5) * 6.0
    poll_d = max(0.30, min(0.72, actual_d + poll_error_pp / 100.0))
    poll_r = round(1.0 - poll_d, 4)
    poll_d = round(poll_d, 4)

    previous_d = max(0.30, min(0.72, actual_d - (rng.random() - 0.5) * 6.0 / 100.0))
    previous_r = round(1.0 - previous_d, 4)
    previous_d = round(previous_d, 4)

    registered_voters = STATE_REGISTERED_VOTERS[state]
    historical_turnout = 0.62 if cycle in {2018, 2020, 2024} else 0.45
    turnout = int(registered_voters * historical_turnout * (0.9 + 0.2 * rng.random()))

    dem_fundraising = int(
        2_500_000 + abs(state_lean_pp) * 80_000 * rng.random() + 1_000_000 * rng.random()
    )
    rep_fundraising = int(
        2_500_000 + abs(state_lean_pp) * 80_000 * rng.random() + 1_000_000 * rng.random()
    )

    race_id = f"US-SEN-{state}-{cycle}"
    return {
        "cycle": cycle,
        "state": state,
        "state_name": STATE_NAMES[state],
        "election_date": ELECTION_DATE[cycle],
        "senate_class": CYCLE_CLASS_LABEL[cycle],
        "dem_name": f"DEM nominee {state} {cycle}",
        "rep_name": f"REP nominee {state} {cycle}",
        "dem_incumbent": incumbent_party == "DEM",
        "rep_incumbent": incumbent_party == "REP",
        "dem_previous_vote_share": previous_d,
        "rep_previous_vote_share": previous_r,
        "dem_fundraising_usd": dem_fundraising,
        "rep_fundraising_usd": rep_fundraising,
        "dem_vote_share": actual_d,
        "rep_vote_share": actual_r,
        "turnout": turnout,
        "winner_party": "DEM" if actual_d > actual_r else "REP",
        "partisan_lean": round(state_lean_pp, 4),
        "incumbency_advantage": incumbent_advantage if incumbent_party != "" else 0.0,
        "economic_index": round(CYCLE_ECONOMY[cycle], 4),
        "demographic_turnout_index": round((rng.random() - 0.5) * 4.0, 4),
        "historical_turnout_rate": historical_turnout,
        "registered_voters": registered_voters,
        "pollster": "State Panel Polling",
        "poll_sample_size": 850,
        "poll_population": "lv",
        "poll_sponsor_class": "nonpartisan",
        "poll_methodology": "mixed",
        "dem_poll_pct": round(poll_d * 100, 4),
        "rep_poll_pct": round(poll_r * 100, 4),
        "race_id": race_id,
        "dem_option_id": f"{race_id}-D",
        "rep_option_id": f"{race_id}-R",
        "incumbent_party": incumbent_party,
        "previous_dem_share": previous_d,
        "previous_rep_share": previous_r,
        "polling_dem_share_base": round(poll_d, 4),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for cycle in CYCLES:
        for state in CYCLE_TO_CLASS[cycle]:
            rows.append(_generate_row(cycle, state))
    rows.sort(key=lambda row: (row["cycle"], row["state"]))

    fieldnames = list(rows[0].keys())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} senate panel rows to {OUTPUT}")


if __name__ == "__main__":
    main()
