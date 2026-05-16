"""Single source of truth for figure dimensions, colors, and CSS.

Every plot in `reports/plots.py` and every HTML page in `reports/diagnostics.py`,
`scoring/cycle_eval.py`, and `reports/race_detail.py` should import from here
so the run dashboards present a coherent visual language.
"""

from __future__ import annotations

from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Canonical figure sizes (W x H inches, 150 dpi assumed).
# ---------------------------------------------------------------------------

SIZE_HERO: Final[tuple[float, float]] = (9.6, 4.0)
"""Full-width, single-figure cards in the hero or section-headline slot."""

SIZE_PANEL: Final[tuple[float, float]] = (6.4, 4.6)
"""Mid-width panels meant to sit two-up in a grid."""

SIZE_RACE: Final[tuple[float, float]] = (4.5, 3.0)
"""Per-race miniatures used in small-multiples grids."""

SIZE_TALL: Final[tuple[float, float]] = (9.6, 6.0)
"""Tall canvases for KDEs, joint plots, and stacked layouts."""

DPI: Final[int] = 150


# ---------------------------------------------------------------------------
# Colour palette.
# ---------------------------------------------------------------------------

PARTY: Final[dict[str, str]] = {
    "DEM": "#2b6cb0",
    "REP": "#c43b3b",
    "IND": "#777777",
    "YES": "#3a8f5d",
    "NO": "#8d8d8d",
}

NEUTRAL: Final[dict[str, str]] = {
    "ink": "#202124",
    "muted": "#656a70",
    "rule": "#d8dde3",
    "bg": "#f6f4ef",
    "card": "#ffffff",
    "panel_bg": "#fbf9f4",
    "axis": "#ffffff",
    "win": "#3a8f5d",
    "loss": "#c43b3b",
}

ACCENT: Final[list[str]] = [
    "#245b8f",
    "#9c6f19",
    "#547c70",
    "#b07aa1",
    "#76b7b2",
    "#c87922",
]


def party_color(party: object, default: str | None = None) -> str:
    """Look up a chamber-friendly party colour with a sensible fallback."""
    key = str(party or "").upper()
    return PARTY.get(key, default or ACCENT[0])


# ---------------------------------------------------------------------------
# Matplotlib defaults.
# ---------------------------------------------------------------------------

_RC_PARAMS: Final[dict[str, object]] = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.facecolor": NEUTRAL["bg"],
    "axes.facecolor": NEUTRAL["axis"],
    "axes.edgecolor": NEUTRAL["rule"],
    "axes.labelcolor": NEUTRAL["ink"],
    "axes.titlecolor": NEUTRAL["ink"],
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": NEUTRAL["rule"],
    "grid.alpha": 0.7,
    "grid.linewidth": 0.7,
    "xtick.color": NEUTRAL["muted"],
    "ytick.color": NEUTRAL["muted"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": ["DejaVu Sans"],
    "font.size": 10,
}

_rc_applied = False


def apply_rcparams() -> None:
    """Install canonical matplotlib defaults exactly once per process."""
    global _rc_applied
    if _rc_applied:
        return
    plt.rcParams.update(_RC_PARAMS)
    _rc_applied = True


def style_axis(
    ax: plt.Axes,
    *,
    grid_axis: str = "y",
    hide: tuple[str, ...] = ("top", "right"),
) -> None:
    """Apply per-axis treatments that aren't expressible via rcParams."""
    for spine in hide:
        ax.spines[spine].set_visible(False)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(axis=grid_axis, color=NEUTRAL["rule"], alpha=0.7, linewidth=0.7)
    else:
        ax.grid(False)


# ---------------------------------------------------------------------------
# Shared CSS for every HTML report.
# ---------------------------------------------------------------------------


def report_css() -> str:
    """Single CSS block used across diagnostics, cycle_eval, and race detail."""
    return """
:root {
  --ink: #202124;
  --muted: #656a70;
  --rule: #d8dde3;
  --bg: #f6f4ef;
  --card: #ffffff;
  --panel-bg: #fbf9f4;
  --dem: #2b6cb0;
  --rep: #c43b3b;
  --win: #3a8f5d;
  --loss: #c43b3b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 36px 32px 64px;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
}
.container { max-width: 1080px; margin: 0 auto; }
header.hero { margin-bottom: 28px; }
.eyebrow {
  margin: 0 0 6px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-size: 11px;
  font-weight: 700;
}
h1 { margin: 0 0 8px; font-size: 40px; line-height: 1.1; letter-spacing: -0.01em; }
h2 { margin: 32px 0 14px; font-size: 22px; }
h3 { margin: 20px 0 10px; font-size: 16px; color: var(--muted); }
.subtitle { color: var(--muted); margin: 4px 0 0; max-width: 720px; }
.section {
  margin-bottom: 28px;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 10px;
  box-shadow: 0 6px 22px rgba(40, 35, 24, .04);
  padding: 22px 24px 24px;
}
.section h2 { margin-top: 0; }
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin: 8px 0 18px;
}
.card {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 14px 16px;
}
.card.hero-card { background: var(--panel-bg); }
.card span.label {
  display: block;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 6px;
  font-weight: 700;
}
.card strong.value {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.card .detail { color: var(--muted); font-size: 12px; margin-top: 4px; }
.plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 18px;
  margin-top: 8px;
}
.plot-grid figure {
  margin: 0;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 12px 12px 14px;
}
.plot-grid figure img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
}
.plot-grid figure figcaption {
  color: var(--muted);
  font-size: 12px;
  margin-top: 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
thead th {
  text-align: left;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .04em;
  border-bottom: 1px solid var(--rule);
  padding: 8px 6px;
  font-weight: 700;
}
tbody td {
  border-bottom: 1px solid var(--rule);
  padding: 9px 6px;
  vertical-align: top;
}
tbody tr:last-child td { border-bottom: none; }
.pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .03em;
}
.pill.dem { background: rgba(43, 108, 176, .12); color: var(--dem); }
.pill.rep { background: rgba(196, 59, 59, .12); color: var(--rep); }
.pill.win { background: rgba(58, 143, 93, .14); color: var(--win); }
.pill.loss { background: rgba(196, 59, 59, .14); color: var(--loss); }
.pill.neutral { background: rgba(101, 106, 112, .12); color: var(--muted); }
a { color: var(--dem); }
a.race-link { font-weight: 600; }
.narrative {
  background: var(--panel-bg);
  border-left: 3px solid var(--dem);
  padding: 12px 16px;
  margin: 10px 0 16px;
  border-radius: 4px;
  font-size: 14px;
  color: var(--ink);
}
.narrative em { color: var(--muted); font-style: normal; }
.audit pre {
  background: #f0eee6;
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 12px 14px;
  font-size: 12px;
  overflow-x: auto;
}
.kpi-row { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-end; }
.kpi-row .kpi { min-width: 140px; }
.kpi span.label {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 700;
  display: block;
}
.kpi strong.value { font-size: 30px; font-weight: 700; letter-spacing: -0.01em; }
.kpi .detail { color: var(--muted); font-size: 12px; }
@media (max-width: 720px) {
  body { padding: 24px 16px 48px; }
  h1 { font-size: 30px; }
  .plot-grid { grid-template-columns: 1fr; }
  .kpi-row { gap: 16px; }
}
"""


# Apply rcparams immediately on import so any plot called via this module
# already runs with the standardized look-and-feel.
apply_rcparams()
