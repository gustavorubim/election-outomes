"""Report generation."""

from election_outcomes.reports.diagnostics import DiagnosticsReport
from election_outcomes.reports.methodology import MethodologySnapshot
from election_outcomes.reports.model_card import ModelCard
from election_outcomes.reports.plots import PlotGenerator
from election_outcomes.reports.silver_benchmark import SilverStyleBenchmark, benchmark_to_json

__all__ = [
    "DiagnosticsReport",
    "MethodologySnapshot",
    "ModelCard",
    "PlotGenerator",
    "SilverStyleBenchmark",
    "benchmark_to_json",
]
