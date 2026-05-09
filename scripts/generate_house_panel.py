"""Generate a synthetic-but-plausible U.S. House district panel for cycles 2014-2024.

Honesty note: rows here are deterministic procedural draws calibrated to roughly
match aggregate historical patterns (D/R seat swings, Cook PVI distribution).
They are NOT actual historical House election results. The panel exists to exercise
the rolling-origin harness at scale; production runs should ingest real returns from
MIT Election Lab or the Daily Kos election data archives.

Two redistricting eras are present:
- 2012_2020 era: cycles 2014, 2016, 2018, 2020 (10-CD-per-state-redistribution from 2010 Census)
- 2022_plus era: cycles 2022, 2024 (post-2020 Census redistricting)

The panel keeps the same 435 generic district codes across both eras to match
fixture pipeline expectations; the redistricting_era column lets downstream
consumers filter rolling-origin training to a single era.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fixtures" / "house_district_panel.csv"

CYCLES: list[int] = [2014, 2016, 2018, 2020, 2022, 2024]
FORECAST_ONLY_CYCLES: set[int] = set()
ELECTION_DATE: dict[int, str] = {
    2014: "2014-11-04",
    2016: "2016-11-08",
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
    2024: "2024-11-05",
}

REDISTRICTING_ERA: dict[int, str] = {
    2014: "2012_2020",
    2016: "2012_2020",
    2018: "2012_2020",
    2020: "2012_2020",
    2022: "2022_plus",
    2024: "2022_plus",
}

# Approximate cycle-level D environment (House popular vote bias).
CYCLE_D_ENVIRONMENT: dict[int, float] = {
    2014: -5.0,  # R wave (House popular vote R+5.7)
    2016: -1.0,  # near tie
    2018: +8.5,  # D wave
    2020: +3.0,  # D edge
    2022: -2.0,  # R edge
    2024: -2.5,  # R edge
}

CYCLE_ECONOMY: dict[int, float] = {
    2014: -0.2,
    2016: -0.1,
    2018: +0.1,
    2020: -0.3,
    2022: -0.4,
    2024: -0.2,
}

# House delegation size per state (435 total). Approximate post-2010 census, used for
# both eras for fixture simplicity.
STATE_DELEGATION: dict[str, int] = {
    "AL": 7,
    "AK": 1,
    "AZ": 9,
    "AR": 4,
    "CA": 52,
    "CO": 8,
    "CT": 5,
    "DE": 1,
    "FL": 28,
    "GA": 14,
    "HI": 2,
    "ID": 2,
    "IL": 17,
    "IN": 9,
    "IA": 4,
    "KS": 4,
    "KY": 6,
    "LA": 6,
    "ME": 2,
    "MD": 8,
    "MA": 9,
    "MI": 13,
    "MN": 8,
    "MS": 4,
    "MO": 8,
    "MT": 2,
    "NE": 3,
    "NV": 4,
    "NH": 2,
    "NJ": 12,
    "NM": 3,
    "NY": 26,
    "NC": 14,
    "ND": 1,
    "OH": 15,
    "OK": 5,
    "OR": 6,
    "PA": 17,
    "RI": 2,
    "SC": 7,
    "SD": 1,
    "TN": 9,
    "TX": 38,
    "UT": 4,
    "VT": 1,
    "VA": 11,
    "WA": 10,
    "WV": 2,
    "WI": 8,
    "WY": 1,
}

# Approximate state lean (used to seed district PVI distribution).
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


def _seeded_random(seed: str) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng


def _district_pvi(state: str, district: int, era: str) -> float:
    """Generate a synthetic Cook PVI for a district. Wider than state lean.

    States with more districts get more variance because gerrymandering and
    geographic sorting produce a wider PVI distribution.
    """
    rng = _seeded_random(f"pvi-{state}-{district}-{era}")
    state_lean = STATE_LEAN[state]
    n = STATE_DELEGATION[state]
    spread = max(8.0, 14.0 - 6.0 / max(n, 1))
    pvi = state_lean + (rng.random() - 0.5) * spread * 2
    # Wave-style sorting: pull extremes further apart to mimic safe-seat dominance.
    if abs(pvi) > 6:
        pvi = pvi * 1.15
    return round(pvi, 2)


def _incumbent_party(state: str, district: int, cycle: int) -> str:
    """Persistent incumbency with occasional flips on big environment swings."""
    rng = _seeded_random(f"hou-incumb-{state}-{district}-{cycle}")
    base_pvi = _district_pvi(state, district, REDISTRICTING_ERA[cycle])
    base_party = "DEM" if base_pvi > 0 else "REP"
    flip_probability = max(0.0, 0.12 - abs(base_pvi) / 90.0)
    if rng.random() < flip_probability:
        return "REP" if base_party == "DEM" else "DEM"
    return base_party


def _generate_row(cycle: int, state: str, district: int) -> dict[str, object]:
    rng = _seeded_random(f"hou-{state}-{district}-{cycle}")
    era = REDISTRICTING_ERA[cycle]
    pvi = _district_pvi(state, district, era)
    incumbent_party = _incumbent_party(state, district, cycle)
    incumbent_advantage = 4.0  # House incumbency historically ~3-5pp
    incumbent_pp = incumbent_advantage if incumbent_party == "DEM" else -incumbent_advantage

    environment_pp = CYCLE_D_ENVIRONMENT[cycle]
    candidate_quality_pp = (rng.random() - 0.5) * 4.0
    expected_d_share_pp = 50.0 + pvi + environment_pp + candidate_quality_pp + incumbent_pp
    actual_d_pp = expected_d_share_pp + (rng.random() - 0.5) * 5.0
    actual_d = max(0.10, min(0.92, actual_d_pp / 100.0))
    actual_r = round(1.0 - actual_d, 4)
    actual_d = round(actual_d, 4)

    competitive = abs(pvi) <= 8 and abs(expected_d_share_pp - 50.0) <= 8
    poll_error_pp = (rng.random() - 0.5) * 7.0
    poll_d = max(0.10, min(0.92, actual_d + poll_error_pp / 100.0))
    poll_r = round(1.0 - poll_d, 4)
    poll_d = round(poll_d, 4)

    previous_d = max(0.10, min(0.92, actual_d - (rng.random() - 0.5) * 6.0 / 100.0))
    previous_r = round(1.0 - previous_d, 4)
    previous_d = round(previous_d, 4)

    # House districts are a fraction of the state — ~700k people each.
    registered_voters = int(550_000 * (0.85 + 0.30 * rng.random()))
    historical_turnout = 0.55 if cycle in {2018, 2020, 2024} else 0.40
    turnout = int(registered_voters * historical_turnout * (0.9 + 0.2 * rng.random()))

    if competitive:
        dem_fundraising = int(800_000 + abs(pvi) * 30_000 + 2_000_000 * rng.random())
        rep_fundraising = int(800_000 + abs(pvi) * 30_000 + 2_000_000 * rng.random())
    else:
        dem_fundraising = int(150_000 + 600_000 * rng.random())
        rep_fundraising = int(150_000 + 600_000 * rng.random())

    district_code = f"{state}-{district:02d}"
    race_id = f"US-HOUSE-{district_code}-{cycle}"
    return {
        "cycle": cycle,
        "state": state,
        "state_name": STATE_NAMES[state],
        "district": district_code,
        "election_date": ELECTION_DATE[cycle],
        "redistricting_era": era,
        "cook_pvi": pvi,
        "competitive": competitive,
        "dem_name": f"DEM nominee {district_code} {cycle}",
        "rep_name": f"REP nominee {district_code} {cycle}",
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
        "partisan_lean": pvi,
        "incumbency_advantage": incumbent_advantage,
        "economic_index": round(CYCLE_ECONOMY[cycle], 4),
        "demographic_turnout_index": round((rng.random() - 0.5) * 4.0, 4),
        "historical_turnout_rate": historical_turnout,
        "registered_voters": registered_voters,
        "pollster": "District Panel Polling",
        "poll_sample_size": 600 if competitive else 400,
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
        for state, delegation_size in STATE_DELEGATION.items():
            for district in range(1, delegation_size + 1):
                rows.append(_generate_row(cycle, state, district))
    rows.sort(key=lambda row: (row["cycle"], row["state"], row["district"]))

    fieldnames = list(rows[0].keys())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    competitive_count = sum(1 for row in rows if row["competitive"])
    print(f"wrote {len(rows)} house panel rows to {OUTPUT} (competitive: {competitive_count})")


if __name__ == "__main__":
    main()
