from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from election_outcomes.config import ProjectContext
from election_outcomes.pipeline import ForecastPipeline

app = typer.Typer(help="U.S. election forecasting engine.")
forecast_app = typer.Typer(help="Forecast commands.")
backtest_app = typer.Typer(help="Backtest commands.")
report_app = typer.Typer(help="Report commands.")
benchmark_app = typer.Typer(help="Performance benchmark commands.")
results_app = typer.Typer(help="Forecast-vs-actual result comparison commands.")
app.add_typer(forecast_app, name="forecast")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(results_app, name="results")
console = Console()


def _context(
    root: Path | None = None,
    sources_config: str = "sources.yaml",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
) -> ProjectContext:
    return ProjectContext.create(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )


@app.command()
def sync(
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
) -> None:
    """Snapshot configured sources into the raw lake."""
    context = _context(root=root, sources_config=sources_config, data_dir=data_dir)
    manifest = ForecastPipeline(context).sync()
    console.print(f"[green]Synced[/green] {manifest.height} sources into {context.raw_dir}")


@app.command("build-features")
def build_features(
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
) -> None:
    """Build curated tables and race tiering."""
    context = _context(root=root, sources_config=sources_config, data_dir=data_dir)
    bundle = ForecastPipeline(context).build_features()
    console.print(f"[green]Built[/green] {bundle.race_catalog.height} race catalog rows")


@forecast_app.command("run")
def forecast_run(
    as_of: str | None = typer.Option(None, help="Forecast date in YYYY-MM-DD form."),
    run_id: str | None = typer.Option(None, help="Stable run id."),
    scenario: str | None = typer.Option(None, help="Scenario key from configs/scenarios.yaml."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Refresh data and emit the full forecast artifact contract."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    out_dir = ForecastPipeline(context).run_forecast(as_of=as_of, run_id=run_id, scenario=scenario)
    console.print(f"[green]Forecast complete[/green]: {out_dir}")


@backtest_app.command("run")
def backtest_run(
    run_id: str | None = typer.Option(None, help="Stable backtest run id."),
    scenario: str | None = typer.Option(None, help="Scenario key from configs/scenarios.yaml."),
    start_cycle: int | None = typer.Option(None, help="First holdout cycle to score."),
    holdout_cycle: int | None = typer.Option(None, help="Single holdout cycle to score."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run historical backtest scorecards and ablations."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_backtest(
        run_id=run_id,
        scenario=scenario,
        start_cycle=start_cycle,
        holdout_cycle=holdout_cycle,
    )
    console.print(f"[green]Backtest complete[/green]: {payload['row_count']} rows")


@report_app.command("build")
def report_build(
    run_id: str = typer.Option(..., help="Forecast run id to rebuild."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Rebuild diagnostics and methodology files for an existing forecast run."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    out_dir = ForecastPipeline(context).rebuild_report(run_id=run_id)
    console.print(f"[green]Report rebuilt[/green]: {out_dir}")


@benchmark_app.command("run")
def benchmark_run(
    as_of: str = typer.Option(..., help="Benchmark date in YYYY-MM-DD form."),
    run_id: str | None = typer.Option(None, help="Stable benchmark run id."),
    draws: int | None = typer.Option(None, help="Override benchmark draw count."),
    repeats: int | None = typer.Option(None, help="Override repeat count."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Benchmark simulation throughput using the configured performance engine."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_benchmark(
        as_of=as_of,
        run_id=run_id,
        draws=draws,
        repeats=repeats,
    )
    console.print(
        "[green]Benchmark complete[/green]: "
        f"{payload['rows_per_second']:.0f} rows/sec via {payload['performance']['engine']}"
    )


@results_app.command("compare")
def results_compare(
    forecast_run_id: str = typer.Option(..., help="Existing forecast run id."),
    comparison_id: str | None = typer.Option(None, help="Stable comparison id."),
    cycle: int | None = typer.Option(None, help="Restrict comparison to a cycle."),
    office_type: str | None = typer.Option(None, help="Restrict comparison to an office type."),
    race_id: str | None = typer.Option(None, help="Restrict comparison to a race id."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Compare a forecast run with known actual results."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).compare_results(
        forecast_run_id=forecast_run_id,
        comparison_id=comparison_id,
        cycle=cycle,
        office_type=office_type,
        race_id=race_id,
    )
    console.print(
        "[green]Comparison complete[/green]: "
        f"{payload['race_count']} races, winner accuracy={payload['winner_accuracy']}"
    )
    console.print(payload["output_dir"])


if __name__ == "__main__":
    app()
