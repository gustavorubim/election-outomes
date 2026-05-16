"""Report generation."""

from civic_signal.reports.diagnostics import DiagnosticsReport
from civic_signal.reports.methodology import MethodologySnapshot
from civic_signal.reports.model_card import ModelCard
from civic_signal.reports.plots import PlotGenerator
from civic_signal.reports.race_detail import RaceDetailRenderer
from civic_signal.reports.silver_benchmark import SilverStyleBenchmark, benchmark_to_json

__all__ = [
    "DiagnosticsReport",
    "MethodologySnapshot",
    "ModelCard",
    "PlotGenerator",
    "RaceDetailRenderer",
    "SilverStyleBenchmark",
    "benchmark_to_json",
]
